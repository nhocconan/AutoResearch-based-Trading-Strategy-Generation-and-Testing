#!/usr/bin/env python3
"""
6h Williams %R + 12h Trend Filter + Volume Confirmation
Hypothesis: Williams %R identifies oversold/overbought conditions on 6s, filtered by 12h EMA trend for direction and volume spikes for confirmation. Works in bull markets via buying oversold dips in uptrends and bear markets via selling overbought bounces in downtrends. Targets 12-30 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_r_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR trend reverses
            if (williams_r[i] > -20 or 
                close[i] < ema_50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR trend reverses
            if (williams_r[i] < -80 or 
                close[i] > ema_50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 12h EMA50
            uptrend = close[i] > ema_50_12h_aligned[i]
            downtrend = close[i] < ema_50_12h_aligned[i]
            
            # Long: Williams %R < -80 (oversold) with uptrend and volume spike
            if (williams_r[i] < -80 and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: Williams %R > -20 (overbought) with downtrend and volume spike
            elif (williams_r[i] > -20 and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals