#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX trend filter and volume confirmation
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to show bull/bear strength.
# We go long when bull power > 0 and rising, short when bear power < 0 and falling,
# confirmed by 1d ADX > 25 (trending market) and volume spike.
# Designed to work in both bull and bear markets by capturing institutional buying/selling pressure.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_ElderRay_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def ema(data, period):
    """Exponential Moving Average"""
    return pd.Series(data).ewm(span=period, adjust=False, min_periods=period).mean().values

def wilders_smoothing(data, period):
    """Wilder's smoothing (used in ADX calculation)"""
    result = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return result
    result[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def adx(high, low, close, period=14):
    """Average Directional Index"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx_vals = wilders_smoothing(dx, period)
    
    return adx_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_14_1d = adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Elder Ray components on 6h data
    ema13 = ema(close, 13)
    bull_power = high - ema13  # High - EMA
    bear_power = low - ema13   # Low - EMA
    
    # Smooth the power indicators to reduce noise
    bull_power_smooth = ema(bull_power, 5)
    bear_power_smooth = ema(bear_power, 5)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_14_1d_aligned[i]
        bull_val = bull_power_smooth[i]
        bear_val = bear_power_smooth[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull power positive AND rising + strong trend (ADX>25) + volume spike
            if (bull_val > 0 and bull_val > bull_power_smooth[i-1] and 
                adx_val > 25 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear power negative AND falling + strong trend (ADX>25) + volume spike
            elif (bear_val < 0 and bear_val < bear_power_smooth[i-1] and 
                  adx_val > 25 and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull power turns negative OR trend weakens
            if bull_val <= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear power turns positive OR trend weakens
            if bear_val >= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals