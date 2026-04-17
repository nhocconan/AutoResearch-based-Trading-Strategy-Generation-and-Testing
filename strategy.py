#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volume spike filter and weekly EMA trend filter.
# In bull markets: breakouts capture momentum. In bear markets: trend filter avoids false breakouts,
# volume spike ensures institutional participation. Target 20-40 trades/year per symbol.
# Position size 0.25 to balance capture and drawdown control.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR volume spike calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period ATR on daily
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume on daily
    avg_vol_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR and average volume to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR for position sizing (volatility normalization)
    tr_4h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_4h[0] = high[0] - low[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(avg_vol_20_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current 1d volume > 1.5x 20-day average
        volume_spike = volume_1d[i // 16] > 1.5 * avg_vol_20[i // 16] if i >= 16 else False
        
        # Trend filter: price above weekly EMA50 for long, below for short
        long_trend = close[i] > ema50_1w_aligned[i]
        short_trend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout above upper band with volume spike and trend alignment
            if long_trend and volume_spike and close[i] > highest_high[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band with volume spike and trend alignment
            elif short_trend and volume_spike and close[i] < lowest_low[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian lower band or reverses at midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < lowest_low[i] or close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian upper band or reverses at midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > highest_high[i] or close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_WeeklyEMA50"
timeframe = "4h"
leverage = 1.0