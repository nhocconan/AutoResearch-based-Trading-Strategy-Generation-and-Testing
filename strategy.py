#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) + 1-week EMA(34) trend filter + volume spike.
# Williams %R identifies overbought/oversold conditions; EMA(34) on weekly defines major trend.
# In uptrend (price > weekly EMA34), buy oversold (W%R < -80); in downtrend (price < weekly EMA34), sell overbought (W%R > -20).
# Volume spike confirms momentum. Designed for 6h to capture medium-term reversals with trend alignment.
# Expects ~20-30 trades/year per symbol, avoiding overtrading.
name = "6h_WilliamsR_WeeklyEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + Williams %R oversold + volume spike
            if (close[i] > ema_34_1w_aligned[i] and 
                williams_r[i] < -80 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + Williams %R overbought + volume spike
            elif (close[i] < ema_34_1w_aligned[i] and 
                  williams_r[i] > -20 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if weekly trend turns down or Williams %R overbought
            if (close[i] < ema_34_1w_aligned[i]) or (williams_r[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if weekly trend turns up or Williams %R oversold
            if (close[i] > ema_34_1w_aligned[i]) or (williams_r[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals