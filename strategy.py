#!/usr/bin/env python3
# 1d_1w_donchian_breakout_v1
# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (EMA50), volume confirmation, and ATR stoploss.
# Long when price breaks above Donchian upper band, 1w EMA50 rising, and volume > 1.5x 20-day average.
# Short when price breaks below Donchian lower band, 1w EMA50 falling, and volume > 1.5x 20-day average.
# Exit when price crosses 10-day EMA in opposite direction.
# Target: 15-25 trades/year with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # 1d Donchian(20) channels
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 1d 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_50_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_10[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below 10-day EMA
            if close[i] < ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above 10-day EMA
            if close[i] > ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian upper, 1w EMA50 rising, volume surge
            if (close[i] > upper[i] and 
                ema_1w_50_aligned[i] > ema_1w_50_aligned[i-1] and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower, 1w EMA50 falling, volume surge
            elif (close[i] < lower[i] and 
                  ema_1w_50_aligned[i] < ema_1w_50_aligned[i-1] and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals