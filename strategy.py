#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with volume confirmation and ADX trend filter.
- Enter long when price breaks above Donchian(20) high + volume > 1.5x 20-period volume MA + ADX(14) > 20
- Enter short when price breaks below Donchian(20) low + volume > 1.5x 20-period volume MA + ADX(14) > 20
- Exit when price crosses back inside Donchian channel
- Fixed position size 0.25 to manage drawdown
- Uses price breakouts with volume confirmation and trend strength to avoid whipsaw
- Designed for 4h timeframe with strict entry conditions to target 75-200 total trades over 4 years
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Trend filter: ADX(14)
    # Calculate True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high).diff()
    down_move = pd.Series(low).diff() * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high.iloc[i]) or np.isnan(donch_low.iloc[i]) or 
            np.isnan(volume_ma_20.iloc[i]) or np.isnan(adx.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper = donch_high.iloc[i]
        lower = donch_low.iloc[i]
        adx_val = adx.iloc[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and trend filter
            # Long: price breaks above upper Donchian + volume spike + ADX > 20
            if price > upper and vol > 1.5 * vol_ma and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume spike + ADX > 20
            elif price < lower and vol > 1.5 * vol_ma and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses back inside Donchian channel
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses back inside Donchian channel
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_VolumeADX"
timeframe = "4h"
leverage = 1.0