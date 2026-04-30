#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
# Long when price breaks above 6h Donchian upper band with 1d low volatility (ATR ratio < 0.8) and volume > 1.5x avg.
# Short when price breaks below 6h Donchian lower band with 1d low volatility (ATR ratio < 0.8) and volume > 1.5x avg.
# ATR regime filter avoids whipsaws in high volatility markets (e.g., 2022 crash, 2025 bear).
# Donchian breakouts capture sustained moves; volatility filter improves win rate in ranging/ bear markets.
# Timeframe: 6h, HTF: 1d for ATR regime, no 1w needed as 1d suffices for regime classification.

name = "6h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 50-period moving average for regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_14_1d / atr_ma_50_1d  # < 1 = low volatility regime
    
    # Align 1d ATR ratio to 6h timeframe (completed 1d bar only)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # warmup for ATR ratio and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr_ratio = atr_ratio_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        
        # Low volatility regime filter: ATR ratio < 0.8 (avoid high vol whipsaws)
        low_vol_regime = curr_atr_ratio < 0.8
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, low vol regime, volume confirmation
            if (curr_close > curr_upper and 
                low_vol_regime and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, low vol regime, volume confirmation
            elif (curr_close < curr_lower and 
                  low_vol_regime and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: price retouches Donchian middle (mean reversion within channel)
            mid = (curr_upper + curr_lower) / 2
            if curr_close <= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price retouches Donchian middle
            mid = (curr_upper + curr_lower) / 2
            if curr_close >= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals