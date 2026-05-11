#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Price breaks above Camarilla R3 or below S3 on 12h, filtered by 1d EMA34 trend and volume spike. Camarilla levels act as dynamic support/resistance derived from prior day's range, effective in ranging and trending markets. Trend filter ensures alignment with longer-term momentum. Volume confirms conviction. Designed for 15-35 trades/year per symbol to minimize fee drag while capturing strong moves in both bull and bear markets.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 1d Camarilla Levels (based on prior day's range) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using prior day's data)
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    hl_range = high_1d - low_1d
    camarilla_r3 = close_1d_prev + 1.1 * hl_range
    camarilla_s3 = close_1d_prev - 1.1 * hl_range
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # --- Volume Filter: spike above 1.5x median of last 30 periods ---
    vol_median = pd.Series(volume_12h).rolling(window=30, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for volume median and EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss (using ATR approximation from 12h range)
                atr_approx = (high_12h[i] - low_12h[i])  # simple range-based volatility
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema34_1d_aligned[i]
        trend_down = close_12h[i] < ema34_1d_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_12h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_12h[i] > camarilla_r3_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above Camarilla R3 + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < camarilla_s3_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below Camarilla S3 + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss (using ATR approximation)
                atr_approx = (high_12h[i] - low_12h[i])
                if close_12h[i] <= entry_price - 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or below Camarilla pivot point (approx. Close)
                elif close_12h[i] <= camarilla_s3_aligned[i] + 0.5 * camarilla_r3_aligned[i] - 0.5 * camarilla_s3_aligned[i]:  # PP ≈ (H+L+C)/3, simplified as midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss (using ATR approximation)
                atr_approx = (high_12h[i] - low_12h[i])
                if close_12h[i] >= entry_price + 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or above Camarilla pivot point
                elif close_12h[i] >= camarilla_s3_aligned[i] + 0.5 * camarilla_r3_aligned[i] - 0.5 * camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals