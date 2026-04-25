#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator crossover with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend direction to align with weekly momentum.
- Williams Alligator: Jaw (13-period SMMA, 8-shift), Teeth (8-period SMMA, 5-shift), Lips (5-period SMMA, 3-shift).
- Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1w EMA50 AND volume > 1.5 * 20-day average volume.
- Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1w EMA50 AND volume > 1.5 * 20-day average volume.
- Exit: Opposite Alligator alignment (Lips crosses Teeth in opposite direction).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to catch strong trends aligned with weekly momentum while filtering chop/whipsaws.
- Works in bull markets (trend continuation up) and bear markets (trend continuation down).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) aka Wilder's MA"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.empty_like(values, dtype=float)
    result[:] = np.nan
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator components (SMMA with shifts)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price = (high + low) / 2
    jaw_raw = smma(median_price, 13)
    jaw = np.roll(jaw_raw, 8)  # Shift right by 8 (look back)
    jaw[:8] = np.nan  # First 8 values invalid after shift
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = smma(median_price, 8)
    teeth = np.roll(teeth_raw, 5)  # Shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = smma(median_price, 5)
    lips = np.roll(lips_raw, 3)  # Shift right by 3
    lips[:3] = np.nan
    
    # Align Alligator components to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # Already LTF, but using for consistency
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Calculate 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # EMA50, VolMA20, Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        jaw_level = jaw_aligned[i]
        teeth_level = teeth_aligned[i]
        lips_level = lips_aligned[i]
        ema_50_level = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips_level > teeth_level and teeth_level > jaw_level
        bearish_alignment = lips_level < teeth_level and teeth_level < jaw_level
        
        # Trend alignment conditions
        above_ema = curr_close > ema_50_level
        below_ema = curr_close < ema_50_level
        
        # Exit conditions: opposite Alligator alignment (Lips crosses Teeth)
        if position != 0:
            # Exit long: bearish alignment (Lips < Teeth)
            if position == 1:
                if lips_level < teeth_level:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish alignment (Lips > Teeth)
            elif position == -1:
                if lips_level > teeth_level:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with trend and volume filters
        if position == 0:
            # Long: bullish alignment AND above EMA50 AND volume confirmation
            long_condition = bullish_alignment and above_ema and volume_confirm
            
            # Short: bearish alignment AND below EMA50 AND volume confirmation
            short_condition = bearish_alignment and below_ema and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0