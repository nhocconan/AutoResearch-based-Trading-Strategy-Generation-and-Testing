#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend
Hypothesis: On 4h timeframe, price breaking above/below Camarilla R1/S1 levels (calculated from prior day) 
confirmed by 1d EMA50 trend direction and volume > 1.5x 20-period average provides edge in both bull and bear markets.
Trades limited to ~20-40 per year via strict entry conditions, avoiding fee drag while capturing strong moves.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d EMA50 for trend direction ---
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Previous day OHLC for Camarilla (R1/S1) ---
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    # First bar uses current day's values
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    r1 = pivot + (range_val * 1.1 / 12.0)
    s1 = pivot - (range_val * 1.1 / 12.0)
    
    # Align to 4h
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- Volume confirmation: current > 1.5x 20-period average ---
    vol_ma20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ok = volume_4h > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start_idx = 50  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                # Simple stop: reverse signal on opposite break
                if position == 1 and close_4h[i] < s1_4h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] > r1_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        if position == 0:
            # Look for breakout entries with trend and volume confirmation
            # Long: break above R1 with uptrend (close > EMA50) and volume spike
            if (close_4h[i] > r1_4h[i] and 
                close_4h[i] > ema50_1d_aligned[i] and 
                vol_ok[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            # Short: break below S1 with downtrend (close < EMA50) and volume spike
            elif (close_4h[i] < s1_4h[i] and 
                  close_4h[i] < ema50_1d_aligned[i] and 
                  vol_ok[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Manage existing position: exit on opposite break or trend reversal
            if position == 1:
                # Long: exit if price breaks below S1 or trend turns down
                if close_4h[i] < s1_4h[i] or close_4h[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit if price breaks above R1 or trend turns up
                if close_4h[i] > r1_4h[i] or close_4h[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals