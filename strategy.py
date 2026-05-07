#!/usr/bin/env python3
name = "6h_1d_LiquiditySweep_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate previous day high and low for liquidity sweep detection
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align daily levels to 6h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price sweeps below previous day low then reverses above it
            # with volume and in daily uptrend
            sweep_low = low[i] < prev_low_aligned[i]
            reclaim = close[i] > prev_low_aligned[i]
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if sweep_low and reclaim and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price sweeps above previous day high then reverses below it
            # with volume and in daily downtrend
            elif (high[i] > prev_high_aligned[i] and 
                  close[i] < prev_high_aligned[i] and 
                  vol_condition and 
                  not uptrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below swing low or volume drops
            swing_low = low[i] < prev_low_aligned[i]
            if swing_low or volume[i] < vol_ma_4[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above swing high or volume drops
            swing_high = high[i] > prev_high_aligned[i]
            if swing_high or volume[i] < vol_ma_4[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h liquidity sweep reversal with daily trend and volume confirmation
# - Looks for price to sweep previous day's high/low (liquidity grab) then reverse
# - Long when sweeps below prior day low then closes back above it with volume in uptrend
# - Short when sweeps above prior day high then closes back below it with volume in downtrend
# - Works in both bull (buy sweeps of lows in uptrend) and bear (sell sweeps of highs in downtrend)
# - Volume confirmation (1.5x average) filters false breaks
# - Exit when price breaks the swing point again or volume drops significantly
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses 6-hour timeframe for better signal quality than lower timeframes
# - Daily trend filter ensures alignment with higher timeframe momentum
# - Designed to exploit stop hunts and market maker behavior in crypto markets