#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and 1w trend filter
# Donchian(20) breakout captures price extremes and momentum
# Volume > 1.3x average confirms breakout strength
# 1w EMA(50) trend filter ensures we trade with higher timeframe trend
# Exit when price crosses opposite Donchian band
# Target: 15-25 trades/year per symbol to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for volume and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume average (20 periods)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d volume and 1w EMA to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian Channel (20 periods) on 12h
    donch_len = 20
    upper_donch = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lower_donch = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, donch_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_donch[i]) or 
            np.isnan(lower_donch[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 1d average
        volume_confirmed = volume[i] > 1.3 * vol_ma_1d_aligned[i]
        
        # Trend filter: price above/below 1w EMA(50)
        price_above_trend = close[i] > ema_50_1w_aligned[i]
        price_below_trend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + volume + uptrend
            if (close[i] > upper_donch[i-1] and 
                volume_confirmed and 
                price_above_trend):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower Donchian + volume + downtrend
            elif (close[i] < lower_donch[i-1] and 
                  volume_confirmed and 
                  price_below_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian
            if close[i] < lower_donch[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above upper Donchian
            if close[i] > upper_donch[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_Volume_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0