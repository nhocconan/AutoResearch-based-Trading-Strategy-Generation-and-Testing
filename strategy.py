#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; 1w HMA ensures alignment with weekly trend
# Volume spike (>2.0 x 20-period EMA) confirms breakout validity
# Works in bull markets (breakout above upper band + uptrend) and bear markets (breakdown below lower band + downtrend)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag

name = "1d_Donchian20_Breakout_1wHMA_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA20 for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w data for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w HMA(21)
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(df_1w['close'].values, half_length)
    wma_full = wma(df_1w['close'].values, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_21_1w = wma(raw_hma, sqrt_length)
    
    # Align HTF indicators to LTF
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and HMA)
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmation = volume[i] > (2.0 * vol_ema_20[i])
        
        # Trend filter from 1w HMA
        uptrend = close[i] > hma_21_1w_aligned[i]
        downtrend = close[i] < hma_21_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper Donchian with volume confirmation and uptrend
            if close[i] > high_20_aligned[i] and volume_confirmation and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower Donchian with volume confirmation and downtrend
            elif close[i] < low_20_aligned[i] and volume_confirmation and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below lower Donchian or trend changes to downtrend
            if close[i] < low_20_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian or trend changes to uptrend
            if close[i] > high_20_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals