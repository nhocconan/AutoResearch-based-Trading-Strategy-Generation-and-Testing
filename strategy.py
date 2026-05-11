#!/usr/bin/env python3
# 1d_Camarilla_R3S3_Breakout_WeeklyTrend
# Hypothesis: Uses weekly trend direction with Camarilla pivot breakout on daily chart.
# Long when: 1) weekly trend is bullish (price > weekly EMA34), 2) price breaks above Camarilla R3 level, 3) volume > 1.5x 20-day average.
# Short when: 1) weekly trend is bearish (price < weekly EMA34), 2) price breaks below Camarilla S3 level, 3) volume > 1.5x 20-day average.
# Exit when price returns to daily EMA34 or weekly trend reverses.
# Weekly trend filter reduces noise and aligns with higher timeframe momentum.
# Camarilla R3/S3 levels provide institutional support/resistance with statistical edge.
# Volume confirmation filters out weak breakouts.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.

name = "1d_Camarilla_R3S3_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly trend: price vs EMA34 ---
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = close_1w > ema34_1w
    weekly_trend_down = close_1w < ema34_1w
    
    # Align weekly trend to daily
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # --- Daily EMA34 for exit ---
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # --- Camarilla pivot levels (based on previous day) ---
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # Using previous day's OHLC to avoid look-ahead
    camarilla_r3 = np.roll(close, 1) + 1.1 * (np.roll(high, 1) - np.roll(low, 1))
    camarilla_s3 = np.roll(close, 1) - 1.1 * (np.roll(high, 1) - np.roll(low, 1))
    # First bar: no previous day
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # --- Volume confirmation (volume > 20-day average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 and volume MA(20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or
            np.isnan(weekly_trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend
        is_weekly_up = weekly_trend_up_aligned[i]
        is_weekly_down = weekly_trend_down_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_weekly_up and vol_spike:
                # Long: weekly uptrend + volume spike + price above Camarilla R3
                if close[i] > camarilla_r3[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_weekly_down and vol_spike:
                # Short: weekly downtrend + volume spike + price below Camarilla S3
                if close[i] < camarilla_s3[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to EMA34 OR weekly trend turns down
                if close[i] < ema34[i] or not is_weekly_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to EMA34 OR weekly trend turns up
                if close[i] > ema34[i] or not is_weekly_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals