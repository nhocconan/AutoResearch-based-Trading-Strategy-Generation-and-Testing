#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Uses 1w EMA200 as trend filter: long only when price > EMA200, short only when price < EMA200
# - Entry: Williams %R(14) crosses above -20 for long or below -80 for short, with volume > 1.5x 20-period average
# - Exit: Williams %R crosses back below -80 for long or above -20 for short
# - Williams %R identifies overbought/oversold conditions for mean reversion
# - 1w trend filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false signals
# - Target: 15-30 trades per year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on 1w data
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Load 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R(14) on 12h data
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_12h = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema_200_12h[i]) or np.isnan(williams_r_12h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price above 1w EMA200 + Williams %R crosses above -20 + volume surge
            if (price > ema_200_12h[i] and 
                williams_r_12h[i] > -20 and 
                williams_r_12h[i-1] <= -20 and 
                vol > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below 1w EMA200 + Williams %R crosses below -80 + volume surge
            elif (price < ema_200_12h[i] and 
                  williams_r_12h[i] < -80 and 
                  williams_r_12h[i-1] >= -80 and 
                  vol > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses back below -80
            if williams_r_12h[i] < -80 and williams_r_12h[i-1] >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses back above -20
            if williams_r_12h[i] > -20 and williams_r_12h[i-1] <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_1wTrendFilter_Volume"
timeframe = "12h"
leverage = 1.0