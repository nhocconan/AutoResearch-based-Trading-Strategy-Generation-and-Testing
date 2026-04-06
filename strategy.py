#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h EMA trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions
# 12h EMA (50) defines trend direction - only trade in direction of trend
# Volume confirmation reduces false signals
# Works in bull/bear by aligning with higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_williamsr_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) - momentum oscillator
    # Values: -100 to 0, oversold below -80, overbought above -20
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    willr = -100 * ( (highest_high - close) / (highest_high - lowest_low + 1e-10) )
    willr = willr.values
    
    # 12h EMA (50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.3 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(willr[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to neutral range (-50) or opposite extreme
        if position == 1:  # long position
            if willr[i] >= -20 or willr[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if willr[i] <= -80 or willr[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries aligned with 12h EMA trend
            # Long: price above 12h EMA (uptrend) AND Williams %R oversold (<-80) + volume
            if (close[i] > ema_12h_aligned[i] and 
                willr[i] < -80 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 12h EMA (downtrend) AND Williams %R overbought (>-20) + volume
            elif (close[i] < ema_12h_aligned[i] and 
                  willr[i] > -20 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals