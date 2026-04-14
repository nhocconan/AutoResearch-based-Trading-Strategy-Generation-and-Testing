#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d ADX Trend Filter and Volume Confirmation
# Williams %R identifies overbought/oversold conditions: readings below -80 = oversold, above -20 = overbought
# Long when: Williams %R crosses above -80 (exit oversold) + ADX > 25 (trending) + volume > 1.5x average
# Short when: Williams %R crosses below -20 (enter overbought) + ADX > 25 (trending) + volume > 1.5x average
# Exit when: Williams %R crosses back through -50 (middle) or volume drops below average
# Designed to capture mean reversion within trending markets, avoiding chop
# Target: 15-35 trades per symbol over 4 years (4-9/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Williams %R (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for Williams %R and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        williams_r_val = williams_r_aligned[i]
        vol_1d_current = volume[i] if i < len(volume) else volume[-1]  # use current 4h volume for spike detection
        
        if position == 0:
            # Long setup: Williams %R crosses above -80 (exit oversold) with volume spike and strong trend
            if (williams_r_val > -80 and williams_r_aligned[i-1] <= -80 and  # crossover up
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and            # Volume spike
                adx_aligned[i] > 25):                                     # Strong trend
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R crosses below -20 (enter overbought) with volume spike and strong trend
            elif (williams_r_val < -20 and williams_r_aligned[i-1] >= -20 and  # crossover down
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and              # Volume spike
                  adx_aligned[i] > 25):                                        # Strong trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses below -50 or volume drops
            if williams_r_val < -50 and williams_r_aligned[i-1] >= -50 or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses above -50 or volume drops
            if williams_r_val > -50 and williams_r_aligned[i-1] <= -50 or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_ADX_Volume"
timeframe = "4h"
leverage = 1.0