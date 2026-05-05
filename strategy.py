#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter and volume confirmation
# Long when 6h Bull Power > 0 AND 1d ADX > 25 (trending) AND 6h close > 6h EMA20 AND volume > 1.5x 20-period average
# Short when 6h Bear Power < 0 AND 1d ADX > 25 (trending) AND 6h close < 6h EMA20 AND volume > 1.5x 20-period average
# Exit when 1d ADX < 20 (range) OR Elder Power reverses sign
# Uses 6h primary timeframe with 1d HTF for ADX regime filter
# Elder Ray captures bull/bear power via EMA13, effective in both bull and bear markets
# ADX regime filter avoids whipsaws in ranging markets
# Volume confirmation ensures momentum behind moves
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_ElderRay_1dADX25_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter (trending when ADX > 25)
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        
        # Directional Movement
        dm_plus = pd.Series(df_1d['high']).diff()
        dm_minus = -pd.Series(df_1d['low']).diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
        
        # Smoothed DM
        dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        
        # Directional Indicators
        di_plus = 100 * (dm_plus_smooth / atr)
        di_minus = 100 * (dm_minus_smooth / atr)
        
        # DX and ADX
        dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
        adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        adx_1d = adx.values
    else:
        adx_1d = np.full(len(df_1d), np.nan)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema_13 = np.full(n, np.nan)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 6h EMA20 for trend filter
    if len(close) >= 20:
        ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        ema_20 = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND ADX > 25 (trending) AND close > EMA20 AND volume spike
            if (bull_power[i] > 0 and 
                adx_1d_aligned[i] > 25 and 
                close[i] > ema_20[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND ADX > 25 (trending) AND close < EMA20 AND volume spike
            elif (bear_power[i] < 0 and 
                  adx_1d_aligned[i] > 25 and 
                  close[i] < ema_20[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 (range) OR Bear Power >= 0 (power reversal)
            if adx_1d_aligned[i] < 20 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX < 20 (range) OR Bull Power <= 0 (power reversal)
            if adx_1d_aligned[i] < 20 or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals