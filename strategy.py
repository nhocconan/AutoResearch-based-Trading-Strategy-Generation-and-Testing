# 12h_1d_volatility_breakout_v2
# 12h timeframe with 1d volatility filter and ATR-based breakout
# Uses volatility regime filter to avoid choppy markets and focus on trending periods
# Works in both bull and bear markets by filtering for high volatility trending periods
# Entry: Price breaks Donchian(20) channel with volatility filter and ATR confirmation
# Exit: Opposite channel touch or volatility regime shift

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility and ATR calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-day ATR on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR with proper handling
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(atr_1d)):
        atr_1d[i] = np.nanmean(tr[i-13:i+1])
    
    # Align ATR to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period volatility (ATR/price ratio) on daily data
    vol_ratio_1d = atr_1d / close_1d
    vol_ratio_12h = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate 20-period moving average of volatility ratio for regime filter
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(vol_ratio_12h[i-20:i])
    
    # Calculate Donchian channels (20-period) on 12h data
    # Highest high and lowest low over last 20 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Channel width for breakout confirmation
    channel_width = highest_high - lowest_low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ratio_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current volatility > 1.2x 20-period average (trending regime)
        vol_filter = vol_ratio_12h[i] > vol_ma[i] * 1.2
        
        # Breakout conditions with ATR confirmation
        # Long breakout: price above upper channel + volatility filter + minimum ATR threshold
        long_breakout = (close[i] > highest_high[i]) and vol_filter and (channel_width[i] > 0.5 * atr_12h[i])
        
        # Short breakout: price below lower channel + volatility filter + minimum ATR threshold
        short_breakout = (close[i] < lowest_low[i]) and vol_filter and (channel_width[i] > 0.5 * atr_12h[i])
        
        # Exit conditions: touch opposite channel or volatility regime shift to choppy
        long_exit = (close[i] < lowest_low[i]) or (vol_ratio_12h[i] < vol_ma[i] * 0.8)
        short_exit = (close[i] > highest_high[i]) or (vol_ratio_12h[i] < vol_ma[i] * 0.8)
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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

name = "12h_1d_volatility_breakout_v2"
timeframe = "12h"
leverage = 1.0