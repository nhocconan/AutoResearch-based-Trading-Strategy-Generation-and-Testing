#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation
Hypothesis: On 12h timeframe, price breaking above Camarilla R1 level with 1d EMA50 uptrend and volume spike signals strong bullish momentum. Conversely, breaking below S1 level with 1d EMA50 downtrend and volume spike signals bearish momentum. Uses discrete sizing (±0.25) and close-based exits to limit trades to 12-37/year, minimizing fee drag while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d average volume for volume confirmation
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate Camarilla levels from previous 1d bar (using 1d OHLC)
    # Camarilla levels are based on previous day's range
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # We need to align these to 12h bars: use previous completed 1d bar's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 12h timeframe (already delayed by align_htf_to_ltf for completed bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    volume_spike = volume > (1.5 * avg_vol_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: need enough for 1d EMA50 (50) + 1d volume avg (20) + alignment buffer
    start_idx = max(50, 20) + 4  # +4 to ensure 1d bar completion (12h -> 1d: 2 bars per day, but use 4 for safety)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = (close_val > r1_level) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < s1_level) and vol_spike and (close_val < ema_50_val)
        
        # Exit conditions: reverse signal or trend deterioration
        long_exit = (close_val < s1_level) or (close_val < ema_50_val)
        short_exit = (close_val > r1_level) or (close_val > ema_50_val)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0