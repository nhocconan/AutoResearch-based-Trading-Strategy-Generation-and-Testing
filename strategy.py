#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; 1w HMA ensures alignment with higher-timeframe trend
# Volume spike (>2.0 x 20-period EMA) confirms breakout validity and reduces false signals
# Works in bull markets (breakout above upper band + uptrend) and bear markets (breakdown below lower band + downtrend)
# Uses discrete position sizing (0.30) to balance return potential and drawdown control
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
    
    # 1d Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1w data for trend filter (HMA21)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate HMA21 on weekly close
    close_1w = df_1w['close'].values
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    # WMA function for HMA calculation
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    wma_half = np.array([wma(close_1w[i:i+half_length], half_length) 
                         if i+half_length <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    wma_full = np.array([wma(close_1w[i:i+21], 21) 
                         if i+21 <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    
    raw_hma = 2 * wma_half - wma_full
    hma_21_1w = np.array([wma(raw_hma[i:i+sqrt_length], sqrt_length) 
                          if i+sqrt_length <= len(raw_hma) else np.nan 
                          for i in range(len(raw_hma))])
    
    # Align HTF HMA to LTF (1d)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and HMA)
    start_idx = max(lookback, 21)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w HMA
        uptrend = close[i] > hma_21_1w_aligned[i]
        downtrend = close[i] < hma_21_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above upper Donchian band with volume confirmation and uptrend
            if high[i] > highest_high[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: Break below lower Donchian band with volume confirmation and downtrend
            elif low[i] < lowest_low[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price closes below mid-point of Donchian channel OR trend changes to downtrend
            mid_point = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] < mid_point or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price closes above mid-point of Donchian channel OR trend changes to uptrend
            mid_point = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] > mid_point or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals