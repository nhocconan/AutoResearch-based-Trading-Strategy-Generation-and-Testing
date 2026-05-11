#!/usr/bin/env python3
"""
12h_1d_Camarilla_R3_S3_Breakout_Trend
Hypothesis: Price breaking above/below Camarilla R3/S3 levels on 12h, filtered by 1d EMA50 trend and volume above median. Exit on opposite Camarilla level or ATR stop. Targets ~20-30 trades/year.
"""

name = "12h_1d_Camarilla_R3_S3_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 12h Camarilla Pivot Levels (from previous 12h candle) ---
    # Calculate pivot from previous candle's OHLC
    prev_close = np.roll(close_12h, 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    # First value will be invalid, but we'll handle with warmup
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 2.0)  # R3 = pivot + 1.1*(H-L)/2
    s3 = pivot - (range_hl * 1.1 / 2.0)  # S3 = pivot - 1.1*(H-L)/2
    
    # --- Volume Filter: above median of last 20 periods ---
    vol_median = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    
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
    
    # Start after warmup period (need enough for EMA50, ATR, volume median)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_median[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_12h[i] <= entry_price - 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema50_1d_aligned[i]
        trend_down = close_12h[i] < ema50_1d_aligned[i]
        
        # Volume filter: above median
        vol_ok = volume_12h[i] > vol_median[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if close_12h[i] > r3[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < s3[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_12h[i] <= entry_price - 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below S3
                elif close_12h[i] <= s3[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_12h[i] >= entry_price + 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above R3
                elif close_12h[i] >= r3[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals