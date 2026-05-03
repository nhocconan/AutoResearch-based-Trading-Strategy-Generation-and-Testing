#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high with volume > 1.5x 24-bar average and close > 1d EMA34 (uptrend)
# Short when price breaks below Donchian(20) low with volume > 1.5x 24-bar average and close < 1d EMA34 (downtrend)
# Exit when price crosses 1d EMA34 in opposite direction or Donchian middle line
# Donchian channels provide clear structure, EMA34 filters trend, volume confirms breakout strength
# Target: 75-200 total trades over 4 years = 19-50/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "4h_Donchian20_Volume_1dEMA34_v1"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle_line = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, lookback, 24) + 1  # EMA34(1d) + Donchian(20) + volume MA(24) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high with volume spike and close > 1d EMA34 (uptrend)
            if (close[i] > highest_high[i] and 
                volume_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume spike and close < 1d EMA34 (downtrend)
            elif (close[i] < lowest_low[i] and 
                  volume_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below 1d EMA34 (trend failure) or crosses below Donchian middle
            if (close[i] < ema_34_aligned[i] or 
                close[i] < middle_line[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA34 (trend failure) or crosses above Donchian middle
            if (close[i] > ema_34_aligned[i] or 
                close[i] > middle_line[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals