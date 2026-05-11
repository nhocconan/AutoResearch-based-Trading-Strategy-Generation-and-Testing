#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
- Long when: close breaks above R1, 1d EMA34 uptrend, volume > 20-period average
- Short when: close breaks below S1, 1d EMA34 downtrend, volume > 20-period average
- Exit when price returns to Camarilla Pivot point or trend reverses
Camarilla levels provide institutional support/resistance. Works in bull by buying breakouts,
in bear by selling breakdowns. Volume filter ensures institutional participation.
Targets 20-30 trades/year (80-120 over 4 years) to minimize fee drag.
"""

name = "4h_12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Camarilla Levels from previous day ---
    # Typical price = (high + low + close) / 3
    typical_price = (high_4h + low_4h + close_4h) / 3.0
    
    # Calculate daily OHLC from 4h data
    # We'll use rolling window of 6 periods (6*4h = 24h) to approximate daily
    # But better: use actual 1d data from df_1d for accuracy
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla for each day, then align to 4h
    # Camarilla levels based on previous day's range
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    # Pivot = (high + low + close) / 3
    prev_day_high = np.concatenate([[daily_high[0]], daily_high[:-1]])
    prev_day_low = np.concatenate([[daily_low[0]], daily_low[:-1]])
    prev_day_close = np.concatenate([[daily_close[0]], daily_close[:-1]])
    
    # Calculate levels for each day
    camarilla_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    camarilla_range = prev_day_high - prev_day_low
    camarilla_R1 = camarilla_pivot + camarilla_range * 1.1 / 12.0
    camarilla_S1 = camarilla_pivot - camarilla_range * 1.1 / 12.0
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40  # for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema34_1d_aligned[i]
        trend_down = close_4h[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if close_4h[i] > R1_aligned[i] and trend_up and vol_ok:
                # Long: break above R1 + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_4h[i] < S1_aligned[i] and trend_down and vol_ok:
                # Short: break below S1 + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to pivot OR trend turns down
                if close_4h[i] <= pivot_aligned[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot OR trend turns up
                if close_4h[i] >= pivot_aligned[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals