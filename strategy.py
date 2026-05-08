#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w ADX trend filter + volume confirmation
# Elder Ray (Bull/Bear Power) measures bullish/bearish pressure relative to EMA.
# Strong trends show sustained Bull/Bear Power with ADX > 25.
# We enter when Bull Power > 0 and rising (for long) or Bear Power < 0 and falling (for short),
# confirmed by 1w ADX > 25 and volume spike (>2x 20-period average).
# Exits occur when power weakens or ADX falls below 20.
# This captures sustained momentum while avoiding whipsaws in low-volatility environments.
# Targets 12-30 trades per year (~48-120 total over 4 years) to minimize fee drag.

name = "6h_ElderRay_1wADX_Volume"
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
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.nansum(arr[:period])  # First value: simple average
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    atr_1w = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w > 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w > 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period)
    
    # Align 1w ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        adx_val = adx_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 and rising, ADX > 25 (strong trend), volume confirmation
            if i > start_idx:
                bull_rising = bull_val > bull_power[i-1]
            else:
                bull_rising = False
            if bull_val > 0 and bull_rising and adx_val > 25 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 and falling, ADX > 25 (strong trend), volume confirmation
            elif i > start_idx:
                bear_falling = bear_val < bear_power[i-1]
            else:
                bear_falling = False
            if bear_val < 0 and bear_falling and adx_val > 25 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or not rising, or ADX < 20 (weakening trend)
            if i > start_idx:
                bull_rising = bull_val > bull_power[i-1]
            else:
                bull_rising = True  # Assume rising on first bar to avoid premature exit
            if bull_val <= 0 or not bull_rising or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or not falling, or ADX < 20 (weakening trend)
            if i > start_idx:
                bear_falling = bear_val < bear_power[i-1]
            else:
                bear_falling = True  # Assume falling on first bar to avoid premature exit
            if bear_val >= 0 or not bear_falling or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals