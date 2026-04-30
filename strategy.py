#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index + 1d ADX regime filter + volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
# 1d ADX > 25 filters for trending markets only (avoids chop)
# Volume confirmation (1.5x 20-period average) ensures momentum strength
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_1dADX25_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 25)
    dm_plus_smooth = wilders_smoothing(dm_plus, 25)
    dm_minus_smooth = wilders_smoothing(dm_minus, 25)
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = wilders_smoothing(dx, 25)
    adx_1d[np.isnan(adx_1d)] = 0
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray components (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Slope of Bull/Bear Power (3-period change)
    bull_power_slope = bull_power - np.roll(bull_power, 3)
    bear_power_slope = bear_power - np.roll(bear_power, 3)
    bull_power_slope[:3] = 0
    bear_power_slope[:3] = 0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 25)  # warmup for EMA13 and 1d ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Regime filter: only trade when 1d ADX > 25 (trending market)
        if adx_1d_aligned[i] <= 25:
            signals[i] = 0.0
            continue
            
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_bull_slope = bull_power_slope[i]
        curr_bear_slope = bear_power_slope[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume confirmation
            if curr_volume_confirm:
                # Long when Bull Power > 0 AND rising AND Bear Power < 0
                if curr_bull_power > 0 and curr_bull_slope > 0 and curr_bear_power < 0:
                    signals[i] = 0.25
                    position = 1
                # Short when Bear Power < 0 AND falling AND Bull Power > 0
                elif curr_bear_power < 0 and curr_bear_slope < 0 and curr_bull_power > 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Bull Power <= 0 OR Bull Power stops rising
            if curr_bull_power <= 0 or curr_bull_slope <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Bear Power >= 0 OR Bear Power stops falling
            if curr_bear_power >= 0 or curr_bear_slope >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals