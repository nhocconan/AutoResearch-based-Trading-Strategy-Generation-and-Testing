#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1w EMA34 trend filter + volume confirmation.
# Williams %R identifies overbought/oversold conditions on 6h chart.
# 1w EMA34 provides strong trend filter - only take longs in weekly uptrend, shorts in weekly downtrend.
# Volume confirmation reduces false signals. Works in bull via buying dips in uptrend
# and in bear via selling rallies in downtrend. Discrete sizing 0.25.
# Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe.

name = "6h_WilliamsR_1wEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R on 6h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1w_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime from weekly EMA34
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume spike in weekly uptrend
            if wr < -80 and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume spike in weekly downtrend
            elif wr > -20 and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R becomes overbought (> -20) OR trend turns down
            if wr > -20 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R becomes oversold (< -80) OR trend turns up
            if wr < -80 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals