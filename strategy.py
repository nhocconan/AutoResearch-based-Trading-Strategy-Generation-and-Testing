#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses 6h timeframe to target 50-150 trades over 4 years (12-37/year) to minimize fee drag
# Donchian channels from 1d provide clear breakout levels with institutional relevance
# 1d EMA50 ensures alignment with medium-term trend for higher probability trades
# Volume spike (2x 20-period average) confirms institutional participation
# Works in bull markets via breakouts and bear markets via fade of false breakouts
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (prior completed 1d bar's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for prior day calculation
        return np.zeros(n)
    
    # Prior completed 1d bar's high/low for Donchian calculation
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    
    # Calculate Donchian levels: upper (20-period high), lower (20-period low)
    # Using prior completed 1d bars to avoid look-ahead
    upper_20 = pd.Series(prior_high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(prior_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].shift(1)).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Align HTF indicators to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper Donchian AND price > 1d EMA50 (bullish trend) AND volume spike
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian AND price < 1d EMA50 (bearish trend) AND volume spike
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below lower Donchian OR below 1d EMA50 (trend change)
            if close[i] < lower_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above upper Donchian OR above 1d EMA50 (trend change)
            if close[i] > upper_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals