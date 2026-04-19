#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout (20-period) + 1w EMA trend filter + volume confirmation.
# Donchian breakout captures breakouts in both bull and bear markets.
# 1w EMA filter ensures alignment with higher timeframe trend.
# Volume confirmation filters false breakouts.
# Designed for 1d timeframe to achieve low trade frequency (7-25/year) with strong risk-adjusted returns.
# Entry: Close breaks above Donchian upper band + close > 1w EMA + volume > 1.5x 20-day avg.
# Exit: Close breaks below Donchian lower band.
# Uses strict conditions to limit trades and avoid overtrading.

name = "1d_Donchian_1wEMA_Volume"
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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close breaks above Donchian upper + close > 1w EMA + volume filter
            if (close[i] > donchian_high[i] and 
                close[i] > ema_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below Donchian lower + close < 1w EMA + volume filter
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if close breaks below Donchian lower
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if close breaks above Donchian upper
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals