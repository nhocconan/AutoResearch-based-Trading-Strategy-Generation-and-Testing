#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power via EMA(13). ADX > 25 indicates trending market.
# In trending markets: buy when Bull Power > 0 and rising, sell when Bear Power < 0 and falling.
# Volume > 1.5x 20-period average confirms institutional participation.
# Works in bull/bear markets by requiring trend presence (ADX filter). Target: 50-150 trades over 4 years.
name = "6h_ElderRay_1dADX25_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = low - ema_13   # Bear Power = Low - EMA(13)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_6h[i]) or np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_6h[i] > 25
        
        # Elder Ray signals: look for momentum in power
        bull_rising = bull_power_6h[i] > bull_power_6h[i-1]
        bear_falling = bear_power_6h[i] < bear_power_6h[i-1]
        
        if position == 0:
            # Long: bull power positive and rising in trending market
            if bull_power_6h[i] > 0 and bull_rising and trending and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bear power negative and falling in trending market
            elif bear_power_6h[i] < 0 and bear_falling and trending and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bull power turns negative or loses momentum
            if bull_power_6h[i] <= 0 or not bull_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bear power turns positive or loses momentum
            if bear_power_6h[i] >= 0 or not bear_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals