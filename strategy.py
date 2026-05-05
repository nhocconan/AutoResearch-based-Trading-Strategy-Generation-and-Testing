#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversion with 1d ADX25 trend filter and volume spike confirmation
# Long when Williams %R(14) < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Short when Williams %R(14) > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Uses 6h primary timeframe with 1d HTF for ADX trend filter and Williams %R calculation
# Williams %R identifies extreme overbought/oversold conditions that often precede reversals in trending markets
# ADX filter ensures we only trade in trending environments where reversals are more likely to sustain
# Volume spike confirmation reduces false signals from low-participation moves
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe

name = "6h_WilliamsR_Extreme_1dADX25_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX trend filter and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    if len(df_1d) >= 14:
        # True Range
        tr1 = df_1d['high'] - df_1d['low']
        tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
        tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
        
        # Directional Movement
        up_move = df_1d['high'] - df_1d['high'].shift(1)
        down_move = df_1d['low'].shift(1) - df_1d['low']
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0))
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0))
        
        # Directional Indicators
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
        adx_1d = adx.values
    else:
        adx_1d = np.full(len(df_1d), np.nan)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Williams %R(14) - no look-ahead
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        williams_r = williams_r.values
    else:
        williams_r = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume spike
            if (williams_r[i] < -80 and 
                adx_1d_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume spike
            elif (williams_r[i] > -20 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (exiting oversold territory)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (exiting overbought territory)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals