#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian(20) breakouts with volume confirmation and ADX trend filter
# - Long when price breaks above weekly Donchian upper with volume > 1.3x 20-day MA and ADX > 25
# - Short when price breaks below weekly Donchian lower with volume > 1.3x 20-day MA and ADX > 25
# - Exit when price crosses weekly Donchian midpoint or reverses with volume confirmation
# - Weekly Donchian provides clean breakout signals with proper trend alignment
# - Volume confirmation ensures breakouts have institutional participation
# - ADX filter ensures we only trade in trending conditions, avoiding chop
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_WeeklyDonchian20_Volume_ADX_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_20)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Daily ADX for trend filter (14-period)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_14 / tr_14
    minus_di = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above weekly Donchian high with volume and trend
            if close[i] > donchian_high[i] and volume_filter[i] and adx[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: break below weekly Donchian low with volume and trend
            elif close[i] < donchian_low[i] and volume_filter[i] and adx[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: cross below midpoint or reverse with volume
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < donchian_low[i] and volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: cross above midpoint or reverse with volume
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > donchian_high[i] and volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals