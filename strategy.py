#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX trend filter + volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and Bear Power < 0 (clear bullish momentum) AND 12h ADX > 25 (strong trend) AND volume > 1.5x 20-period average
# Short when Bear Power > 0 and Bull Power < 0 (clear bearish momentum) AND 12h ADX > 25 AND volume > 1.5x 20-period average
# Exit when Elder Bull/Bear power converges (signals weaken) or ADX drops below 20 (trend weakening)
# Uses 12h ADX for trend strength filtering to avoid whipsaws in ranging markets
# Targets 15-35 trades per year for low fee drag (< 140 total over 4 years)

name = "6h_ElderRay_12hADX_Volume"
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
    
    # Elder Ray components: Bull Power and Bear Power (13-period EMA)
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema13 = ema(close, 13)
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 12h ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
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
            # Enter long: Bull Power positive, Bear Power negative (bullish momentum) 
            # AND strong trend (ADX > 25) AND volume confirmation
            if bull_val > 0 and bear_val < 0 and adx_val > 25 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power positive, Bull Power negative (bearish momentum)
            # AND strong trend (ADX > 25) AND volume confirmation
            elif bear_val > 0 and bull_val < 0 and adx_val > 25 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Momentum weakening (Bear Power >= 0 or Bull Power <= 0) 
            # OR trend weakening (ADX < 20)
            if bear_val >= 0 or bull_val <= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Momentum weakening (Bull Power >= 0 or Bear Power <= 0)
            # OR trend weakening (ADX < 20)
            if bull_val >= 0 or bear_val <= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals