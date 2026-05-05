#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND 1d close > 1d EMA34 AND volume > 1.5 * 20-period average volume
# Short when Williams %R > -20 (overbought) AND 1d close < 1d EMA34 AND volume > 1.5 * 20-period average volume
# Williams %R identifies exhaustion points; EMA34 filters trend direction; volume confirms participation.
# Works in bull markets via longs on pullbacks and bear markets via shorts on rallies.
# 12h timeframe reduces trade frequency to minimize fee drag while capturing medium-term reversals.

name = "12h_WilliamsR_Extreme_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Williams %R (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Oversold: Williams %R < -80, Overbought: Williams %R > -20
    oversold = williams_r < -80
    overbought = williams_r > -20
    
    # Align Williams %R signals to prices timeframe
    oversold_aligned = align_htf_to_ltf(prices, df_12h, oversold.astype(float))
    overbought_aligned = align_htf_to_ltf(prices, df_12h, overbought.astype(float))
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Calculate 20-period average volume for volume confirmation
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_filter = volume > (1.5 * avg_vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(oversold_aligned[i]) or np.isnan(overbought_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold AND 1d uptrend AND high volume
            if (oversold_aligned[i] > 0.5 and 
                uptrend_1d_aligned[i] > 0.5 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought AND 1d downtrend AND high volume
            elif (overbought_aligned[i] > 0.5 and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (neutral) OR 1d trend changes to downtrend
            if (overbought_aligned[i] > 0.5 or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (neutral) OR 1d trend changes to uptrend
            if (oversold_aligned[i] > 0.5 or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals