#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and trend filter using 1d EMA34.
# In bull markets, price breaks above upper Donchian channel; in bear markets, breaks below lower channel.
# Volume surge confirms breakout strength. 1d EMA34 filter ensures alignment with higher timeframe trend.
# Designed for 4h timeframe to capture medium-term trends with low frequency (~20-40 trades/year).
# Entry: Long when close > upper Donchian(20) and volume > 1.5x 20-period average and close > 1d EMA34.
# Short when close < lower Donchian(20) and volume > 1.5x 20-period average and close < 1d EMA34.
# Exit: Opposite Donchian touch or trend reversal (close crosses 1d EMA34).
# Uses strict conditions to limit trades and avoid overtrading.
name = "4h_Donchian20_Volume_EMA34Trend"
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
    
    # Upper and lower Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian with volume surge and uptrend
            if (close[i] > high_roll[i] and 
                volume_surge[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume surge and downtrend
            elif (close[i] < low_roll[i] and 
                  volume_surge[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches lower Donchian or trend turns down
            if (close[i] < low_roll[i]) or (close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches upper Donchian or trend turns up
            if (close[i] > high_roll[i]) or (close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals