#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
# Bull Power = EMA(13) - Low, Bear Power = High - EMA(13). 
# Long when Bull Power > 0 and rising + ADX > 20 (trending) + volume spike.
# Short when Bear Power < 0 and falling + ADX > 20 + volume spike.
# Uses 13-period EMA for sensitivity, 1d ADX for regime, volume > 1.5x EMA(20) for confirmation.
# Designed to capture momentum in both bull and bear markets while avoiding whipsaws in ranges.
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag.
name = "6h_ElderRay_1dADX20_VolumeConfirm"
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period ADX for daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Plus/Minus Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    plus_di = 100 * np.where(atr > 0,
                             np.convolve(plus_dm, np.ones(period)/period, mode='full')[:len(plus_dm)] / atr, 0)
    minus_di = 100 * np.where(atr > 0,
                              np.convolve(minus_dm, np.ones(period)/period, mode='full')[:len(minus_dm)] / atr, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0,
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < period:
            adx[i] = np.nan
        elif i == period:
            adx[i] = np.mean(dx[1:period+1])
        else:
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # 6h EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = EMA(13) - Low, Bear Power = High - EMA(13)
    bull_power = ema13 - low
    bear_power = high - ema13
    
    # Rising/Falling power (1-period change)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_falling = bear_power < np.roll(bear_power, 1)
    bull_power_rising[0] = False
    bear_power_falling[0] = False
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(adx_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0 and rising + ADX > 20 + volume spike
            if (bull_power[i] > 0 and bull_power_rising[i] and 
                adx_aligned[i] > 20 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 and falling + ADX > 20 + volume spike
            elif (bear_power[i] < 0 and bear_power_falling[i] and 
                  adx_aligned[i] > 20 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 or ADX drops below 15
            if bull_power[i] <= 0 or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 or ADX drops below 15
            if bear_power[i] >= 0 or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals