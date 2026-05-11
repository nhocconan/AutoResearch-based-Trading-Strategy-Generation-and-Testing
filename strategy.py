#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol
Hypothesis: Use 4h trend (EMA50) and 1d volume spike for signal direction, 1h for entry timing at Camarilla R1/S1 levels.
Breakouts only in direction of 4h trend with 1d volume confirmation. Avoids counter-trend trades to reduce whipsaw.
Designed for low trade frequency (15-30/year) to avoid fee drag while capturing high-probability momentum moves.
Works in bull markets via trend-following breakouts and in bear markets via short breakdowns with volume confirmation.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h EMA50 for trend filter ---
    close_4h = df_4h['close']
    ema_50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- Daily Camarilla Pivot Levels (R1, S1) ---
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)  # R1
    s1 = pivot - (range_1d * 1.1 / 12)  # S1
    
    # Align to 1h (Camarilla levels are valid for the entire day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- 1d Volume Spike (2x 20-period EMA) ---
    vol_ema_1d = df_1d['volume'].ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (2.0 * vol_ema_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Determine 4h trend based on price vs EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # Breakout signals (price crosses R1/S1 with 1d volume spike)
        long_breakout = (high[i] > r1_aligned[i]) and vol_spike_1d_aligned[i]
        short_breakout = (low[i] < s1_aligned[i]) and vol_spike_1d_aligned[i]
        
        if position == 0:
            if price_above_ema:
                # Uptrend: only long breakouts
                if long_breakout:
                    signals[i] = 0.20
                    position = 1
            elif price_below_ema:
                # Downtrend: only short breakouts
                if short_breakout:
                    signals[i] = -0.20
                    position = -1
            # No counter-trend trades to reduce whipsaw
        else:
            # Exit conditions: reverse signal or trend change
            if position == 1:
                # Exit long: short breakdown or trend turns down
                exit_signal = short_breakout or (close[i] < ema_50_4h_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: long breakout or trend turns up
                exit_signal = long_breakout or (close[i] > ema_50_4h_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals