#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX(14) trend filter
# Long when price breaks above Donchian upper band AND volume > 1.5x 20-period average AND 1w ADX > 25 (trending)
# Short when price breaks below Donchian lower band AND volume > 1.5x 20-period average AND 1w ADX > 25 (trending)
# Exit when price crosses back to Donchian midpoint OR 1w ADX drops below 20 (trend weakening)
# Uses discrete sizing (0.30) to limit fee drag. Target: 20-40 trades/year per symbol.
# Donchian channels provide clear structure, volume spike confirms participation, ADX filters for trending markets only.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_VolumeSpike_1wADX_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h data (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate ADX on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    if len(high_1w) >= 30:
        # True Range
        tr1 = pd.Series(high_1w).diff().abs()
        tr2 = (pd.Series(high_1w) - pd.Series(close_1w).shift()).abs()
        tr3 = (pd.Series(low_1w) - pd.Series(close_1w).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        
        # Directional Movement
        up_move = pd.Series(high_1w).diff()
        down_move = -pd.Series(low_1w).diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        
        # Trend filter: ADX > 25 for strong trend, ADX < 20 for weakening trend
        strong_trend = adx > 25
        weakening_trend = adx < 20
    else:
        strong_trend = np.zeros(len(close_1w), dtype=bool)
        weakening_trend = np.zeros(len(close_1w), dtype=bool)
    
    # Align HTF data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, prices, donchian_high)  # Already 4h
    donchian_low_aligned = align_htf_to_ltf(prices, prices, donchian_low)    # Already 4h
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)   # Already 4h
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter.astype(float))
    strong_trend_aligned = align_htf_to_ltf(prices, df_1w, strong_trend.astype(float))
    weakening_trend_aligned = align_htf_to_ltf(prices, df_1w, weakening_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(volume_filter_aligned[i]) or 
            np.isnan(strong_trend_aligned[i]) or 
            np.isnan(weakening_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND strong trend
            if (close[i] > donchian_high_aligned[i] and 
                volume_filter_aligned[i] > 0.5 and 
                strong_trend_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND strong trend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_filter_aligned[i] > 0.5 and 
                  strong_trend_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian mid OR trend weakening
            if (close[i] < donchian_mid_aligned[i] or 
                weakening_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back to Donchian mid OR trend weakening
            if (close[i] > donchian_mid_aligned[i] or 
                weakening_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals