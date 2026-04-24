#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power (Bull/Bear) with 12h ADX regime filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h ADX for regime (trending if ADX > 25, ranging if ADX < 20) with hysteresis.
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low.
- Entry: Long when Bull Power > 0 AND 12h ADX > 25 (trending up) AND volume > 1.5 * volume MA(20).
         Short when Bear Power > 0 AND 12h ADX > 25 (trending down) AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when Bull Power <= 0,
        exit short when Bear Power <= 0.
- Signal size: 0.25 discrete to balance return and drawdown.
Uses 12h ADX regime filter to avoid whipsaws in ranging markets and Elder Ray to measure bull/bear power relative to EMA13.
Works in both bull (trend following) and bear (trend continuation) markets by only taking trades in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX regime filter and EMA13 for Elder Ray
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate 12h EMA13 for Elder Ray (using typical price)
    typical_price = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3
    ema13 = pd.Series(typical_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 12h ADX for regime filter
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
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_12h - ema13
    bear_power = ema13 - low_12h
    
    # Align HTF indicators to 6h
    ema13_aligned = align_htf_to_ltf(prices, df_12h, ema13)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    adx_state = 0  # 0: undefined, 1: trending (ADX > 25), -1: ranging (ADX < 20)
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Need enough bars for ADX and EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Update ADX regime state with hysteresis
        if adx_aligned[i] > 25:
            adx_state = 1  # Trending
        elif adx_aligned[i] < 20:
            adx_state = -1  # Ranging
        # Else maintain previous state (hysteresis)
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and regime filter
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Bull Power > 0 AND trending up (ADX > 25) AND volume confirmed
            if bull_power_aligned[i] > 0 and adx_state == 1 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND trending down (ADX > 25) AND volume confirmed
            elif bear_power_aligned[i] > 0 and adx_state == 1 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Bull Power <= 0 (bull power fading)
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                adx_state = 0  # Reset regime on exit
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Bear Power <= 0 (bear power fading)
            if bear_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                adx_state = 0  # Reset regime on exit
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_12hADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0