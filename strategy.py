#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_Trend
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) with breakout confirmation and 1-day EMA trend filter on 12h timeframe.
Trades breakouts in trending markets (EMA34) and avoids mean-reversion to reduce trade frequency.
Designed for low trade frequency (<30/year) to avoid fee drag while capturing high-probability moves.
Works in both bull and bear markets by following trend direction via EMA filter.
"""

name = "12h_1d_Camarilla_R1_S1_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    # Align to 12h (Camarilla levels are valid for the entire day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- Volume Spike Detection (2x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ema.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on price vs EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Breakout signals (price crosses R1/S1 with volume spike)
        long_breakout = (high[i] > r1_aligned[i]) and vol_spike[i]
        short_breakout = (low[i] < s1_aligned[i]) and vol_spike[i]
        
        if position == 0:
            if price_above_ema:
                # Uptrend: favor long breakouts, avoid shorts
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
            elif price_below_ema:
                # Downtrend: favor short breakouts, avoid longs
                if short_breakout:
                    signals[i] = -0.25
                    position = -1
            # No mean reversion to reduce trade frequency
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price breaks below EMA or touches S1 (stop/reversal)
                exit_signal = (close[i] < ema_34_1d_aligned[i]) or (low[i] <= s1_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above EMA or touches R1 (stop/reversal)
                exit_signal = (close[i] > ema_34_1d_aligned[i]) or (high[i] >= r1_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals