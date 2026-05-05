#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ATR-based volatility filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND ATR(12h)/ATR(6h) > 1.2 AND volume > 1.3x 20-period average
# Short when price breaks below 20-period Donchian low AND ATR(12h)/ATR(6h) > 1.2 AND volume > 1.3x 20-period average
# Exit when price crosses back to 10-period Donchian midpoint OR ATR ratio < 1.0 (low volatility)
# Donchian channels provide clear structure with proven edge in crypto
# 12h/6h ATR ratio filters for expanding volatility environments (avoids choppy markets)
# Volume confirmation ensures breakout has participation
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 6h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "6h_Donchian20_ATRratio_Volume_Breakout"
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
    
    # Get 12h data ONCE before loop for ATR calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ATR on 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation for 12h
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr1[0]  # First value
    
    # ATR(12h) with 14-period
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR on 6h timeframe (primary)
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]  # First value
    
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 12h ATR to 6h timeframe
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate Donchian channels on 6h (20-period high/low, 10-period midpoint)
    if n >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volatility filter: ATR(12h) / ATR(6h) > 1.2 (expanding volatility)
    vol_ratio = atr_12h_aligned / atr_6h
    vol_filter = vol_ratio > 1.2
    
    # Volume confirmation: volume > 1.3x 20-period average
    if n >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.3 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ratio[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND expanding volatility AND volume spike
            if (close[i] > donchian_high[i] and 
                vol_filter[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND expanding volatility AND volume spike
            elif (close[i] < donchian_low[i] and 
                  vol_filter[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR low volatility environment
            if (close[i] < donchian_mid[i] or 
                vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR low volatility environment
            if (close[i] > donchian_mid[i] or 
                vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals