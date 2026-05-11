#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Touch_Volume_Spike_Dyn_v2
Hypothesis: Uses Camarilla pivot levels (R1, S1) from 1d combined with volume spikes.
Long when price touches S1 with volume spike in bullish regime (price > 1d EMA34).
Short when price touches R1 with volume spike in bearish regime (price < 1d EMA34).
Volume spike defined as current volume > 2.0 * 20-period volume average.
Designed for low trade frequency (~20-30 trades/year) by requiring precise pivot touches
and volume confirmation, reducing false signals. Works in both bull and bear markets
by adapting to the 1d trend regime.
"""

name = "4h_Camarilla_Pivot_Touch_Volume_Spike_Dyn_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Calculate Camarilla pivot levels (R1, S1) from 1d data ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R1 = close_1d + (range_1d * 1.1 / 12)
    S1 = close_1d - (range_1d * 1.1 / 12)
    
    # --- 1d EMA34 for trend filter ---
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # --- Volume spike detection (20-period average) ---
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- Align 1d indicators to 4h ---
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_34_aligned[i]):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for long setup: price touches or goes below S1 with volume spike in bullish regime
            long_setup = (low[i] <= S1_aligned[i]) and volume_spike[i] and (close[i] > ema_34_aligned[i])
            # Look for short setup: price touches or goes above R1 with volume spike in bearish regime
            short_setup = (high[i] >= R1_aligned[i]) and volume_spike[i] and (close[i] < ema_34_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to pivot zone or opposite touch with volume
            if position == 1:
                # Exit long: price touches or goes above R1, or returns to S1 area
                exit_signal = (high[i] >= R1_aligned[i]) or (low[i] >= S1_aligned[i] and close[i] > S1_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches or goes below S1, or returns to R1 area
                exit_signal = (low[i] <= S1_aligned[i]) or (high[i] <= R1_aligned[i] and close[i] < R1_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals