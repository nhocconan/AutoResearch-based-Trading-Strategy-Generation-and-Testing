#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme + 1d EMA200 trend filter + volume spike confirmation
    # Williams %R < -80 = oversold (long), > -20 = overbought (short) on 6h
    # Only trade in direction of 1d EMA200 to avoid counter-trend whipsaws
    # Volume > 1.5x 20-period average confirms momentum
    # Designed for low frequency (target: 12-25/year) to minimize fee drag in 6h timeframe
    # Works in bull/bear markets by only trading with the dominant daily trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Williams %R(14) on 6h
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 6h volume for confirmation
    vol_ma_6h = np.full(len(df_6h), np.nan)
    for i in range(20, len(df_6h)):
        vol_ma_6h[i] = np.mean(volume_6h[i-20:i])
    
    # Volume confirmation: volume > 1.5 * 20-period average (6h)
    volume_spike_6h = volume_6h > (1.5 * vol_ma_6h)
    
    # Align all indicators to LTF
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_6h, volume_spike_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # 1d trend filter
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Extreme %R + trend alignment + volume confirmation
        long_entry = False
        short_entry = False
        
        # Long: oversold (%R < -80) + bullish daily trend + volume spike
        if oversold and bullish_trend:
            long_entry = volume_spike_aligned[i]
        # Short: overbought (%R > -20) + bearish daily trend + volume spike
        elif overbought and bearish_trend:
            short_entry = volume_spike_aligned[i]
        
        # Exit logic: %R returns to neutral zone (-50) or trend changes
        long_exit = (williams_r_aligned[i] > -50) or not bullish_trend
        short_exit = (williams_r_aligned[i] < -50) or not bearish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williams_r_extreme_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0