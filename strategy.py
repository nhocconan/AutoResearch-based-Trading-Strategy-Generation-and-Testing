#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams %R extreme + 1w EMA50 trend filter + volume confirmation
    # Williams %R < -80 = oversold (long), > -20 = overbought (short)
    # Only trade with weekly trend to avoid counter-trend whipsaws
    # Volume spike (>2.0x 20-period average) confirms institutional participation
    # Designed for very low frequency (target: 15-25/year) to minimize fee drag in 1d timeframe
    # Works in bull/bear markets by only trading with the dominant weekly trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume for confirmation
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Volume confirmation: volume > 2.0 * 20-period average (1d)
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Align all indicators to LTF
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # 1w trend filter
        bullish_trend = close[i] > ema50_1w_aligned[i]
        bearish_trend = close[i] < ema50_1w_aligned[i]
        
        # Entry logic: Extreme %R + trend alignment + volume confirmation
        long_entry = False
        short_entry = False
        
        # Long: oversold (%R < -80) + bullish weekly trend + volume spike
        if oversold and bullish_trend:
            long_entry = volume_spike_aligned[i]
        # Short: overbought (%R > -20) + bearish weekly trend + volume spike
        elif overbought and bearish_trend:
            short_entry = volume_spike_aligned[i]
        
        # Exit logic: %R returns to neutral zone (-50) or trend reversal
        long_exit = williams_r_aligned[i] > -50 or not bullish_trend
        short_exit = williams_r_aligned[i] < -50 or not bearish_trend
        
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

name = "1d_1w_williams_r_extreme_trend_volume_v1"
timeframe = "1d"
leverage = 1.0