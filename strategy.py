#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX regime filter
# Long when price breaks above 20-period Donchian high AND volume > 1.5x 20-period average AND ADX(14) > 25
# Short when price breaks below 20-period Donchian low AND volume > 1.5x 20-period average AND ADX(14) > 25
# Exit when price crosses the 20-period Donchian midpoint (mean reversion) OR ADX < 20 (range regime)
# Uses 4h primary timeframe with discrete sizing (0.30) to limit fee drag
# Donchian channels provide clear trend-following structure with proven edge in crypto
# Volume confirmation reduces false breakouts, ADX filter ensures trending conditions
# Target: 75-150 total trades over 4 years (19-37/year) for 4h timeframe

name = "4h_Donchian20_Breakout_Volume_ADX_Trend"
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
    
    # Calculate 20-period Donchian channels
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate ADX(14) for regime filter
    if len(high) >= 14:
        # True Range
        tr1 = pd.Series(high).diff().abs()
        tr2 = pd.Series(low).diff().abs()
        tr3 = abs(pd.Series(high).shift(1) - pd.Series(low).shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Movement
        up_move = pd.Series(high).diff()
        down_move = -pd.Series(low).diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Regime filters
        adx_trend = adx > 25      # Strong trend
        adx_range = adx < 20      # Range market
    else:
        adx = np.full(n, np.nan)
        adx_trend = np.zeros(n, dtype=bool)
        adx_range = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter[i]) or 
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout + volume + trend regime
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                adx_trend[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: Donchian breakdown + volume + trend regime
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  adx_trend[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Donchian mean reversion OR range regime
            if close[i] < donchian_mid[i] or adx_range[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Donchian mean reversion OR range regime
            if close[i] > donchian_mid[i] or adx_range[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals