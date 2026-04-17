#!/usr/bin/env python3
"""
4h Williams %R + Volume Spike + ADX Trend Filter
Long: Williams %R < -80 (oversold) + volume > 1.5x 4h volume SMA(20) + ADX(14) > 20
Short: Williams %R > -20 (overbought) + volume > 1.5x 4h volume SMA(20) + ADX(14) > 20
Exit: Williams %R crosses above -50 (long) or below -50 (short)
Williams %R identifies momentum extremes, volume confirms participation, ADX ensures trending environment.
Designed to work in both bull and bear markets by fading extremes in the direction of the trend.
Target: 80-160 total trades over 4 years (20-40/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.fillna(-50).values  # neutral when undefined
    
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
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume SMA(20)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Williams %R, ADX, and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_sma[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        adx_val = adx[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        
        if position == 0:
            # Long: Williams %R oversold + volume spike + ADX > 20
            if wr < -80 and vol > 1.5 * vol_sma_val and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought + volume spike + ADX > 20
            elif wr > -20 and vol > 1.5 * vol_sma_val and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0