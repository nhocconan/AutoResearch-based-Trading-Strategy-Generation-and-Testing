#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R with 1-day EMA(13) trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions that often precede reversals.
# The 1-day EMA(13) provides a reliable trend filter to ensure trades align with the daily trend.
# Volume > 1.5x the 20-period average confirms institutional participation and reduces false signals.
# Exit occurs when Williams %R returns to neutral territory (above -20 for longs, below -80 for shorts).
# This combination aims for 20-30 trades per year per symbol (80-120 total over 4 years), staying within the optimal range to minimize fee drift.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(13) for trend filter
    ema_len = 13
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams %R (14 periods) on 4h
    williams_len = 14
    if len(high) < williams_len:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=williams_len, min_periods=williams_len).max().values
    lowest_low = pd.Series(low).rolling(window=williams_len, min_periods=williams_len).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, williams_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA13
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + above 1d EMA + volume
            if (williams_r[i] < -80 and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R overbought (> -20) + below 1d EMA + volume
            elif (williams_r[i] > -20 and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (above -20)
            if williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (below -80)
            if williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_EMA13_WilliamsR_Volume_v1"
timeframe = "4h"
leverage = 1.0