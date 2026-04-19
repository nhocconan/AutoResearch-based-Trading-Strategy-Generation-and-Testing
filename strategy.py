#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend alignment using 4h EMA34 for trend direction,
# 1h Donchian15 breakout for momentum, and volume confirmation. Enters only during 08-20 UTC session.
# Uses strict entry conditions to limit trades to 15-35/year (60-140 total over 4 years).
# Works in bull/bear by following 4h trend and avoiding choppy markets with volume filter.
name = "1h_4h_EMA34_Donchian15_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA34 trend (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1h data for Donchian15 breakout (called ONCE before loop)
    high_1h = high
    low_1h = low
    high_15_1h = pd.Series(high_1h).rolling(window=15, min_periods=15).max().values
    low_15_1h = pd.Series(low_1h).rolling(window=15, min_periods=15).min().values
    
    # Volume filter: volume > 1.8 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(high_15_1h[i]) or 
            np.isnan(low_15_1h[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 4h EMA34 AND breaks 1h Donchian high with volume
            if (close[i] > ema_34_4h_aligned[i] and 
                close[i] > high_15_1h[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA34 AND breaks 1h Donchian low with volume
            elif (close[i] < ema_34_4h_aligned[i] and 
                  close[i] < low_15_1h[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 4h EMA34 or 1h Donchian low
            if close[i] < ema_34_4h_aligned[i] or close[i] < low_15_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above 4h EMA34 or 1h Donchian high
            if close[i] > ema_34_4h_aligned[i] or close[i] > high_15_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals