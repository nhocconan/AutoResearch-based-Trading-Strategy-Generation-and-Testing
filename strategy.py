#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator + Elder Ray Power Index with 1-day ADX filter.
# The Williams Alligator (JAWS/TEETH/LIPS) identifies trend direction and strength.
# Elder Ray Power Index measures bull/bear power relative to EMA13.
# 1-day ADX > 20 filters for trending markets, avoiding chop.
# Long: LIPS > TEETH > JAWS (bullish alignment) + Bull Power > 0 + ADX > 20.
# Short: LIPS < TEETH < JAWS (bearish alignment) + Bear Power < 0 + ADX > 20.
# Exit when Alligator alignment breaks or ADX falls below 15.
# This combines trend-following with oscillator confirmation for robust signals in both bull and bear markets.
# Target: 20-30 trades per year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1-day data ONCE for ADX filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1-day ADX(14) for trend strength filter
    adx_len = 14
    if len(df_1d) < adx_len:
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, adx_len)
    dm_plus_smooth = wilders_smoothing(dm_plus, adx_len)
    dm_minus_smooth = wilders_smoothing(dm_minus, adx_len)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, adx_len)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 4h: SMMA(13,8,5) with offsets
    jaw_len, teeth_len, lips_len = 13, 8, 5
    jaw_offset, teeth_offset, lips_offset = 8, 5, 3
    
    def smoothed_moving_average(data, period):
        sma = np.full_like(data, np.nan)
        if len(data) < period:
            return sma
        sma[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smoothed_moving_average(close, jaw_len)
    teeth = smoothed_moving_average(close, teeth_len)
    lips = smoothed_moving_average(close, lips_len)
    
    # Apply offsets (shift right)
    jaw = np.roll(jaw, jaw_offset)
    teeth = np.roll(teeth, teeth_offset)
    lips = np.roll(lips, lips_offset)
    # Set offset periods to NaN
    jaw[:jaw_offset] = np.nan
    teeth[:teeth_offset] = np.nan
    lips[:lips_offset] = np.nan
    
    # Elder Ray Power Index: EMA13 and Bull/Bear Power
    ema_len = 13
    ema_13 = pd.Series(close).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, jaw_len + jaw_offset, teeth_len + teeth_offset, lips_len + lips_offset, ema_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: JAWS (slowest), TEETH (middle), LIPS (fastest)
        bullish_alignment = (lips[i] > teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i] < jaw[i])
        
        # Elder Ray: Bull Power > 0 (bulls in control), Bear Power < 0 (bears in control)
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # ADX filter: trending market
        strong_trend = adx_aligned[i] > 20
        weak_trend = adx_aligned[i] < 15  # Exit when trend weakens
        
        if position == 0:
            # Enter long: bullish alignment + bull power + strong trend
            if bullish_alignment and bull_power_positive and strong_trend:
                position = 1
                signals[i] = position_size
            # Enter short: bearish alignment + bear power + strong trend
            elif bearish_alignment and bear_power_negative and strong_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: alignment breaks or trend weakens
            if not bullish_alignment or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: alignment breaks or trend weakens
            if not bearish_alignment or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Alligator_ElderRay_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0