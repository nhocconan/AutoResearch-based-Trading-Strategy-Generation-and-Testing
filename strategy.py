#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d ADX regime filter and volume confirmation
# Elder Ray uses Bull Power (High - EMA13) and Bear Power (Low - EMA13) to measure bull/bear strength.
# We go long when Bull Power > 0 and Bear Power < 0 (bullish divergence) with ADX > 25 (trending)
# and volume > 1.5x 20-period average. Short when Bear Power < 0 and Bull Power > 0 (bearish divergence).
# Designed to work in both bull and bear markets by capturing strong directional moves.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_ElderRay_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def ema(data, period):
    """Exponential Moving Average"""
    return pd.Series(data).ewm(span=period, adjust=False, min_periods=period).mean().values

def adx(high, low, close, period):
    """Average Directional Index"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(high[i-1] - low[i], 0)
        if plus_dm[i] > minus_dm[i]:
            minus_dm[i] = 0
        elif minus_dm[i] > plus_dm[i]:
            plus_dm[i] = 0
        else:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_vals = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    return adx_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_14_1d = adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Elder Ray components on 6h data
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_14_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, ADX > 25, volume filter
            if (bull_val > 0 and bear_val < 0 and 
                adx_val > 25 and vol_filt):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, Bull Power > 0, ADX > 25, volume filter
            elif (bear_val < 0 and bull_val > 0 and 
                  adx_val > 25 and vol_filt):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or Bear Power >= 0 or ADX < 20
            if not (bull_val > 0 and bear_val < 0) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or Bull Power <= 0 or ADX < 20
            if not (bull_val > 0 and bear_val < 0) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals