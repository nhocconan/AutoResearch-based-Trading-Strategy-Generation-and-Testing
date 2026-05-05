#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme reversal with 1d volume confirmation and 4h ADX trend filter
# Long when Williams %R < -80 (oversold) AND 1d volume > 1.5x 20-period average AND 4h ADX > 25 (trending)
# Short when Williams %R > -20 (overbought) AND 1d volume > 1.5x 20-period average AND 4h ADX > 25 (trending)
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short)
# Uses 4h primary timeframe with 1d HTF for volume confirmation and 4h ADX for trend filter
# Williams %R identifies exhaustion points; volume confirms conviction; ADX ensures trending environment
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_WilliamsR_Extreme_1dVolume_4hADX25"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d volume spike filter
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_filter_1d = vol_1d > (1.5 * vol_ma_20)
    else:
        volume_filter_1d = np.zeros(len(df_1d), dtype=bool)
    
    # Get 4h data ONCE before loop for Williams %R and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Williams %R (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    
    # Calculate 4h ADX (14-period) for trend filter
    # ADX requires +DI, -DI, and TR
    tr1 = pd.Series(high_4h).rolling(window=1).apply(lambda x: x[0]) - pd.Series(low_4h).rolling(window=1).apply(lambda x: x[0])
    tr2 = abs(pd.Series(high_4h).rolling(window=1).apply(lambda x: x[0]) - pd.Series(close_4h).shift(1).rolling(window=1).apply(lambda x: x[0]))
    tr3 = abs(pd.Series(low_4h).rolling(window=1).apply(lambda x: x[0]) - pd.Series(close_4h).shift(1).rolling(window=1).apply(lambda x: x[0]))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_dm = pd.Series(high_4h).diff()
    minus_dm = pd.Series(low_4h).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d volume filter to 4h timeframe
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # Align 4h indicators to 4h timeframe (same df_4h)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND volume spike AND ADX > 25 (trending)
            if (williams_r_aligned[i] < -80 and 
                volume_filter_1d_aligned[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND volume spike AND ADX > 25 (trending)
            elif (williams_r_aligned[i] > -20 and 
                  volume_filter_1d_aligned[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (recovery from oversold)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (recovery from overbought)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals