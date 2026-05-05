#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
# Exit when Williams %R returns to -50 (mean reversion) OR ADX < 20 (trend weakens)
# Uses 6h primary timeframe with 1d HTF for trend filter to avoid counter-trend trades
# Williams %R captures short-term extremes; ADX filters for trending markets; volume confirms conviction
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag

name = "6h_WilliamsR_EXTREME_1dADX25_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data for trend filter
    # ADX requires +DI, -DI, and TR calculations
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = pd.Series(df_1d['low']).diff().abs()
        tr3 = (pd.Series(df_1d['close']).shift(1) - pd.Series(df_1d['high'])).abs()
        tr4 = (pd.Series(df_1d['close']).shift(1) - pd.Series(df_1d['low'])).abs()
        tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        
        # +DM and -DM
        up_move = pd.Series(df_1d['high']).diff()
        down_move = pd.Series(df_1d['low']).diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed +DM, -DM, TR
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
        tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
        
        # +DI and -DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
        
        # Align ADX to 6h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # Calculate Williams %R on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if (williams_r[i] < -80 and 
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif (williams_r[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) OR ADX < 20 (trend weakens)
            if williams_r[i] >= -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) OR ADX < 20 (trend weakens)
            if williams_r[i] <= -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals