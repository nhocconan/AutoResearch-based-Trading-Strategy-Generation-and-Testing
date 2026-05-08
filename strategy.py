#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX trend filter + volume confirmation
# Long when Bull Power > 0, ADX > 25 (trending), and volume > 1.5x 20-period average
# Short when Bear Power < 0, ADX > 25 (trending), and volume > 1.5x 20-period average
# Exit when Elder Power reverses sign or ADX < 20 (range) or volume drops
# Targets 12-37 trades per year (50-150 total over 4 years) for optimal fee drag
# Elder Ray measures bull/bear strength relative to EMA, ADX filters for trending markets,
# volume confirms momentum strength. Works in both bull (strong bull power) and bear (strong bear power) markets.

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
    
    # Elder Ray components (13-period EMA as base)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 12h data for ADX filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- (14-period)
    def smooth_series(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = smooth_series(tr, 14)
    dm_plus_14 = smooth_series(dm_plus, 14)
    dm_minus_14 = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_series(dx, 14)
    
    # Align 12h indicators to 6s timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        adx_val = adx_12h_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, ADX > 25 (trending), volume confirmation
            if bull_val > 0 and adx_val > 25 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, ADX > 25 (trending), volume confirmation
            elif bear_val < 0 and adx_val > 25 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or ADX < 20 (range) or volume drops
            if bull_val <= 0 or adx_val < 20 or not vol_conf_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or ADX < 20 (range) or volume drops
            if bear_val >= 0 or adx_val < 20 or not vol_conf_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals