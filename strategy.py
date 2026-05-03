#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high with volume > 1.8x 20-bar average and close > 1d EMA50 (uptrend)
# Short when price breaks below Donchian(20) low with volume > 1.8x 20-bar average and close < 1d EMA50 (downtrend)
# Exit when price crosses the opposite Donchian band (long exits at 20-low, short exits at 20-high)
# Donchian channels provide clear structure with defined risk; volume confirms breakout strength; 1d EMA50 filters for higher timeframe trend alignment.
# Target: 75-200 total trades over 4 years = 19-50/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "4h_Donchian20_Volume_1dEMA50_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 50) + 1  # Donchian(20) + EMA50(1d)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian(20) high with volume spike and close > 1d EMA50 (uptrend)
            if (close[i] > highest_20[i-1] and  # Break above previous period's high
                volume_spike[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian(20) low with volume spike and close < 1d EMA50 (downtrend)
            elif (close[i] < lowest_20[i-1] and  # Break below previous period's low
                  volume_spike[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian(20) low
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian(20) high
            if close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals