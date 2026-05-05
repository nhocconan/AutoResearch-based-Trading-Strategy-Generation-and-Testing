#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter
# Long when price breaks above 12h Camarilla R3 AND 1d volume > 2.0x 20-period average AND 1d ADX > 25
# Short when price breaks below 12h Camarilla S3 AND 1d volume > 2.0x 20-period average AND 1d ADX > 25
# Exit when price crosses 12h EMA34 (trend reversal)
# Uses 12h primary timeframe with 1d HTF for volume/ADX filters and Camarilla structure
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) based on proven Camarilla breakout performance
# Camarilla levels provide institutional structure; volume confirms breakout validity; ADX ensures trending market
# Works in both bull and bear markets by following the 12h trend while using 1d for filters

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADX25_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume, ADX, and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h close for trend filter and exit
    ema_34_12h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d ADX for regime filter
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Movement
        up = pd.Series(df_1d['high']).diff()
        down = pd.Series(df_1d['low']).diff().abs()
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        adx = np.full(len(df_1d), np.nan)
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 20-period average on 1d
    if len(df_1d) >= 20:
        vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
        volume_filter = df_1d['volume'].values > (2.0 * vol_ma_20)
    else:
        volume_filter = np.full(len(df_1d), False)
    
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    # Calculate Camarilla levels on 1d data (based on previous day's OHLC)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r3 = c_1d + 1.1 * (h_1d - l_1d) * 1.1 / 4
    camarilla_s3 = c_1d - 1.1 * (h_1d - l_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_12h[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND ADX > 25
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter_aligned[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND ADX > 25
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter_aligned[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA34 (trend reversal)
            if close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA34 (trend reversal)
            if close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals