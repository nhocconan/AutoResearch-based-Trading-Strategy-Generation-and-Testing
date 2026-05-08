#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d trend filter
# Long: price breaks above Donchian(20) high + volume > 1.5x average + 1d EMA(50) up
# Short: price breaks below Donchian(20) low + volume > 1.5x average + 1d EMA(50) down
# Exit: opposite Donchian break or volume < 0.5x average
# Uses price structure (Donchian) + volume confirmation + trend filter for robustness
# Targets 20-50 trades/year to minimize fee drag

name = "4h_Donchian20_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: break above Donchian high + volume spike + 1d uptrend
            if (close[i] > donch_high[i-1] and 
                volume[i] > 1.5 * vol_ma[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.30
                position = 1
            # Enter short: break below Donchian low + volume spike + 1d downtrend
            elif (close[i] < donch_low[i-1] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: break below Donchian low or volume drops
            if close[i] < donch_low[i-1] or volume[i] < 0.5 * vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: break above Donchian high or volume drops
            if close[i] > donch_high[i-1] or volume[i] < 0.5 * vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals