// src/reuseable/CandleChart.jsx
import React, { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";

const CandleChart = ({ candles, height = 520 }) => {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { color: "#020617" }, // Deep navy aesthetic
        textColor: "#d1d4dc",
      },
      rightPriceScale: {
        borderColor: "#334155",
      },
      timeScale: {
        borderColor: "#334155",
        rightOffset: 0,
        barSpacing: 6,
      },
      grid: {
        vertLines: { color: "#1e293b" },
        horzLines: { color: "#1e293b" },
      },
    });

    const series = chart.addCandlestickSeries({
      upColor: "#22c55e",     // Tailwind emerald-500
      downColor: "#ef4444",   // Tailwind red-500
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    chartRef.current = chart;
    seriesRef.current = series;

    const handleResize = () => {
      if (!containerRef.current || !chartRef.current) return;
      chartRef.current.applyOptions({
        width: containerRef.current.clientWidth,
        height,
      });
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, [height]);

  // Update data when candles change
  useEffect(() => {
    if (!seriesRef.current || !Array.isArray(candles) || candles.length === 0) return;

    seriesRef.current.setData(candles);

    if (chartRef.current) {
      const ts = chartRef.current.timeScale();

      // Ensure ALL candles show on screen
      const first = candles[0].time;
      const last = candles[candles.length - 1].time;

      ts.setVisibleRange({ from: first, to: last });
    }
  }, [candles]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: `${height}px`,
        borderRadius: "14px",
        overflow: "hidden",
      }}
    />
  );
};

export default CandleChart;