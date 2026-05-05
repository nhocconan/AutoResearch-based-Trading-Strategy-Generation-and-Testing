#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 AND volume > 1.8x 20-period average
# Short when Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 AND volume > 1.8x 20-period average
# Exit when Elder Power signals reverse OR 1d ADX < 20 (regime shift to ranging)
# Elder Ray measures bull/bear strength via EMA(13): Bull Power = High - EMA, Bear Power = Low - EMA
# ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges
# Volume confirmation reduces false breakouts
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 6h (primary), HTF: 1d

name = "6h_ElderRay_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA(13) for Elder Ray
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema_13 = np.full(n, np.nan)
    
    # Calculate Elder Ray components: Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1d ADX(14)
    if len(high_1d) >= 14 and len(low_1d) >= 14 and len(close_1d) >= 14:
        # True Range
        tr1 = np.abs(np.diff(high_1d))
        tr2 = np.abs(np.diff(low_1d))
        tr3 = np.abs(np.diff(close_1d))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        # Pad first element
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = np.diff(high_1d)
        down_move = -np.diff(low_1d)  # positive when low decreases
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        # Pad first element
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = WilderSmoothing(tr, 14)
        plus_di = 100 * WilderSmoothing(plus_dm, 14) / atr
        minus_di = 100 * WilderSmoothing(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = WilderSmoothing(dx, 14)
    else:
        adx = np.full(len(close_1d), np.nan)
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation on 6h (threshold: 1.8x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.8 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Elder Power reverses OR ADX < 20 (ranging market)
            if bull_power[i] <= 0 or bear_power[i] >= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Power reverses OR ADX < 20 (ranging market)
            if bear_power[i] >= 0 or bull_power[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals