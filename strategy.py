#!/usr/bin/env python3
"""
6h_Weekly_Camarilla_R4S4_Breakout_1dTrend_Volume
Hypothesis: Price breaking above/below R4/S4 weekly Camarilla levels on 6h, filtered by 1d EMA50 trend and volume spike (2x median). Uses R4/S4 for strong breakouts only. Weekly structure provides strong support/resistance. Trend filter from daily timeframe ensures alignment. Volume confirms conviction. Designed to work in bull (uptrend breaks) and bear (downtrend breaks). Target: 15-35 trades/year to avoid fee drag.
"""

name = "6h_Weekly_Camarilla_R4S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get weekly data for Camarilla levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Weekly Camarilla Levels (based on previous week) ---
    # Calculate from previous weekly bar (shifted by 1 to avoid lookahead)
    prev_close_w = np.roll(df_w['close'].values, 1)
    prev_high_w = np.roll(df_w['high'].values, 1)
    prev_low_w = np.roll(df_w['low'].values, 1)
    prev_close_w[0] = df_w['close'].values[0]
    prev_high_w[0] = df_w['high'].values[0]
    prev_low_w[0] = df_w['low'].values[0]
    
    # Weekly range
    weekly_range = prev_high_w - prev_low_w
    
    # Camarilla R4 and S4 levels (strong breakout levels)
    camarilla_r4 = prev_close_w + weekly_range * 1.1 / 2
    camarilla_s4 = prev_close_w - weekly_range * 1.1 / 2
    
    # Align weekly levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_w, camarilla_s4)
    
    # --- Volume Filter: spike above 2x median of last 28 periods (approx 1 week) ---
    vol_median = pd.Series(volume_6h).rolling(window=28, min_periods=14).median().values
    vol_threshold = vol_median * 2.0
    
    # --- ATR for stoploss (21-period) ---
    tr1 = np.abs(high_6h - low_6h)
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 60  # for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_6h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_6h[i] > ema50_1d_aligned[i]
        trend_down = close_6h[i] < ema50_1d_aligned[i]
        
        # Volume filter: spike above 2x median
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_6h[i] > camarilla_r4_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above R4 + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            elif close_6h[i] < camarilla_s4_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below S4 + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_6h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below S4
                elif close_6h[i] <= camarilla_s4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_6h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above R4
                elif close_6h[i] >= camarilla_r4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals