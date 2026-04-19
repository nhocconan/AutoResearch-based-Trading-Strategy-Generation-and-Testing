#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ADX(14) trend filter
# Long when price breaks above upper band with volume > 1.5x average and ADX > 25
# Short when price breaks below lower band with volume > 1.5x average and ADX > 25
# Exit when price returns to middle band (20-period average of high/low)
# Uses 4h as primary timeframe with ADX confirming trend strength
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years)

name = "4h_DonchianBreakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min()
    middle = (high_max + low_min) / 2
    
    # Calculate ADX(14) for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need Donchian and ADX data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max.iloc[i]) or np.isnan(low_min.iloc[i]) or 
            np.isnan(adx.iloc[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_max.iloc[i]
        lower = low_min.iloc[i]
        mid = middle.iloc[i]
        adx_val = adx.iloc[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper band with volume and trend
            if price > upper and volume_confirmed and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band with volume and trend
            elif price < lower and volume_confirmed and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to middle band
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle band
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals