#!/usr/bin/env python3
name = "6h_1d_FisherReversal_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Daily Fisher Transform (10-period)
    hlc3 = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    max_hlc3 = hlc3.rolling(window=10, min_periods=10).max()
    min_hlc3 = hlc3.rolling(window=10, min_periods=10).min()
    range_hlc3 = max_hlc3 - min_hlc3
    value1 = np.where(range_hlc3 != 0, 2 * ((hlc3 - min_hlc3) / range_hlc3 - 0.5), 0)
    value1 = np.clip(value1, -0.999, 0.999)
    fish = 0.5 * np.log((1 + value1) / (1 - value1))
    fish = np.where(np.isnan(fish), 0, fish)
    fish_signal = np.where(np.isnan(fish), 0, fish)
    fish_smoothed = pd.Series(fish_signal).ewm(alpha=0.5, adjust=False).mean().values
    
    # Align Fisher to 6h timeframe
    fish_aligned = align_htf_to_ltf(prices, df_1d, fish_smoothed)
    
    # Daily trend filter: EMA(50) on daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(fish_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 with volume and daily uptrend
            fish_cross_up = fish_aligned[i] > -1.5 and fish_aligned[i-1] <= -1.5
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if fish_cross_up and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below 1.5 with volume and daily downtrend
            elif fish_aligned[i] < 1.5 and fish_aligned[i-1] >= 1.5 and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Fisher crosses below -1.5 or volume drops
            if fish_aligned[i] < -1.5 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Fisher crosses above 1.5 or volume drops
            if fish_aligned[i] > 1.5 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Fisher Transform reversal with daily trend and volume confirmation
# - Fisher Transform identifies extreme price reversals (crosses -1.5/1.5)
# - Daily EMA(50) filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) filters false signals
# - Works in both bull (buy Fisher crosses up in uptrend) and bear (sell Fisher crosses down in downtrend)
# - Exit when Fisher reverses or volume weakens
# - Position size 0.25 targets ~30-50 trades/year, avoiding fee drag
# - Fisher Transform is effective in ranging and trending markets, suitable for 2025 bearish conditions