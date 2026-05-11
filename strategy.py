#!/usr/bin/env python3
"""
12h_1w_Camarilla_R3_S3_Breakout_Trend
Hypothesis: Breakout above/below weekly Camarilla R3/S3 levels filtered by 1d EMA34 trend and volume above 70th percentile. Exit on opposite Camarilla touch or ATR stop. Targets ~20-30 trades/year.
"""

name = "12h_1w_Camarilla_R3_S3_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- Weekly Camarilla Levels (R3, S3) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot and ranges
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_r3_1w = close_1w + (range_1w * 1.1 / 4)
    camarilla_s3_1w = close_1w - (range_1w * 1.1 / 4)
    
    # Align weekly levels to 12h timeframe
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    # --- Daily Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Volume Filter: above 70th percentile of last 28 periods ---
    vol_percentile = pd.Series(volume_12h).rolling(window=28, min_periods=14).quantile(0.7).values
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 34  # for EMA34, ATR, volume percentile
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_1w_aligned[i]) or np.isnan(camarilla_s3_1w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_percentile[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema34_1d_aligned[i]
        trend_down = close_12h[i] < ema34_1d_aligned[i]
        
        # Volume filter: above 70th percentile
        vol_ok = volume_12h[i] > vol_percentile[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if close_12h[i] > camarilla_r3_1w_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above weekly Camarilla R3 + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < camarilla_s3_1w_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below weekly Camarilla S3 + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_12h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below weekly Camarilla S3
                elif close_12h[i] <= camarilla_s3_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_12h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above weekly Camarilla R3
                elif close_12h[i] >= camarilla_r3_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals