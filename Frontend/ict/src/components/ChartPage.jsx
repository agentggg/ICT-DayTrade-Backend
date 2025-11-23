// src/pages/ChartPage.jsx
import React, { useEffect, useState, useCallback } from "react";
import CandleChart from "../reuseable/CandleChart";
import getIpAddress from "../config";
import axios from "axios";
import "./ChartPage.css";

const WINDOW = 200;
const ipAddress = getIpAddress();

const TIMEFRAMES = ["1m", "5m", "15m", "1h"];

const ChartPage = () => {
  const [candles, setCandles] = useState([]);
  const [timeframe, setTimeframe] = useState("5m");
  const [isLoading, setIsLoading] = useState(false);
  const [checking, setChecking] = useState(null); // "FVG" | "Bullish OB" | "Bearish OB" | null
  const [lastResult, setLastResult] = useState(null);

  const fetchCandles = useCallback(
    async (tf = timeframe) => {
      try {
        setIsLoading(true);

        const res = await fetch(
          `${ipAddress}/fvg_data/?instrument=GBPUSD&timeframe=${tf}`
        );
        const data_res = await res.json();

        const flat = Array.isArray(data_res[0]) ? data_res.flat() : data_res;
        const slice = flat.length <= WINDOW ? flat : flat.slice(-WINDOW);

        const mapped = slice.map((c) => ({
          time: Math.floor(new Date(c.timestamp).getTime() / 1000),
          open: c._open,
          high: c.high,
          low: c.low,
          close: c.close,
        }));

        setCandles(mapped);
        setTimeframe(tf);
      } catch (err) {
        console.error("Error fetching candles", err);
      } finally {
        setIsLoading(false);
      }
    },
    [timeframe]
  );

  useEffect(() => {
    fetchCandles("5m");
  }, [fetchCandles]);

  const handleClick = async (type, direction) => {
    if (!candles.length) return;

    try {
      if (type === "fvg") {
        setChecking("FVG");
        const response = await axios.post(`${ipAddress}/check_fvg/`, {
          candles,
        });
        const isFvg = !!response.data.is_fvg;
        setLastResult(`FVG · ${isFvg ? "YES ✅" : "NO ❌"}`);
        alert(isFvg ? "Yes, this is an FVG ✅" : "No FVG here ❌");
      } else if (type === "ob") {
        const label =
          direction === "bullish" ? "Bullish Order Block" : "Bearish Order Block";
        setChecking(label);

        const payload = {
          direction,
          candles,
        };

        const res = await axios.post(`${ipAddress}/check_order_block/`, payload);
        // backend returns { ok: true/false }
        const isOb = res.data.ok ?? res.data.is_order_block ?? false;

        setLastResult(`${label} · ${isOb ? "YES ✅" : "NO ❌"}`);
        alert(isOb ? `Yes, valid ${label} ✅` : `No ${label} here ❌`);
      }
    } catch (err) {
      console.error("Check error:", err);
    } finally {
      setChecking(null);
    }
  };

  const refreshPage = () => {
    fetchCandles(timeframe);
  };

  return (
    <div className="chart-root">
      <div className="chart-shell animate-fade-in">
        <header className="chart-header">
          <div>
            <h1 className="chart-title">GBPUSD – 5m Candles</h1>
            <p className="chart-subtitle">ICT Training · Synthetic Feed</p>
          </div>
          <div className="chart-header-right">
            <span className="chip chip-gradient">GBPUSD</span>
            <span className="chip chip-soft">
              {candles.length ? `${candles.length} candles` : "Loading…"}
            </span>
          </div>
        </header>

        <div className="chart-toolbar">
          <div className="timeframe-group">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                className={`tf-btn ${tf === timeframe ? "tf-btn-active" : ""}`}
                onClick={() => fetchCandles(tf)}
              >
                {tf}
              </button>
            ))}
          </div>

          <div className="toolbar-right">
            {isLoading && <span className="pulse-dot" />}
            <button className="icon-btn" onClick={refreshPage} title="Refresh candles">
              ↻
            </button>
          </div>
        </div>

        <div className="chart-card animate-slide-up">
          {isLoading && (
            <div className="chart-loading-overlay">
              <div className="spinner" />
              <span>Updating candles…</span>
            </div>
          )}
          <CandleChart
            candles={candles}
            height={560}          // <- bump this up if you want it even taller
            title="GBPUSD"
            timeframe={timeframe}
            onTimeframeChange={() => {}}
            />
        </div>

        <div className="actions-row animate-slide-up-delayed">
          <button
            className={`action-btn action-fvg ${
              checking === "FVG" ? "action-btn-loading" : ""
            }`}
            onClick={() => handleClick("fvg")}
            disabled={checking !== null}
          >
            {checking === "FVG" ? "Checking FVG…" : "Check FVG"}
          </button>

          <button
            className={`action-btn action-bull ${
              checking === "Bullish Order Block" ? "action-btn-loading" : ""
            }`}
            onClick={() => handleClick("ob", "bullish")}
            disabled={checking !== null}
          >
            {checking === "Bullish Order Block"
              ? "Checking Bullish OB…"
              : "Bullish Order Block"}
          </button>

          <button
            className={`action-btn action-bear ${
              checking === "Bearish Order Block" ? "action-btn-loading" : ""
            }`}
            onClick={() => handleClick("ob", "bearish")}
            disabled={checking !== null}
          >
            {checking === "Bearish Order Block"
              ? "Checking Bearish OB…"
              : "Bearish Order Block"}
          </button>
        </div>

        {lastResult && (
          <div className="result-banner animate-fade-in">
            <span className="result-label">Last Check</span>
            <span className="result-text">{lastResult}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChartPage;