#!/usr/bin/env python3
"""
4h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence/presence on 4h.
When Alligator is "sleeping" (jaw>teeth>lips or lips>teeth>jaw) we avoid trades.
When "awakening" (teeth crosses jaw/lips) we trade in direction of cross with 1d EMA50 trend filter and volume spike.
Designed for 4h timeframe to target 20-50 trades/year. Works in bull markets via trend continuation
and in bear markets via filtering out false signals during chop/sleeping periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 4h: SMAs of median price
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    median_price_4h = (df_4h['high'].values + df_4h['low'].values) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) aka Wilder's MA"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_4h = smma(median_price_4h, 13)
    teeth_4h = smma(median_price_4h, 8)
    lips_4h = smma(median_price_4h, 5)
    
    # Apply shifts: jaw shifted 8, teeth shifted 5, lips shifted 3
    jaw_4h = np.roll(jaw_4h, 8)
    teeth_4h = np.roll(teeth_4h, 5)
    lips_4h = np.roll(lips_4h, 3)
    # Set shifted values to NaN
    jaw_4h[:8] = np.nan
    teeth_4h[:5] = np.nan
    lips_4h[:3] = np.nan
    
    jaw_4h_aligned = align_htf_to_ltf(prices, df_4h, jaw_4h)
    teeth_4h_aligned = align_htf_to_ltf(prices, df_4h, teeth_4h)
    lips_4h_aligned = align_htf_to_ltf(prices, df_4h, lips_4h)
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss on primary timeframe
    if len(close) >= 14:
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(n)
        atr[:13] = np.nan
        for i in range(13, n):
            atr[i] = np.mean(tr[i-13:i+1])
    else:
        atr = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Alligator, EMA50_1d, ATR, and volume MA to propagate
    start_idx = max(50, 13, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_4h_aligned[i]) or 
            np.isnan(teeth_4h_aligned[i]) or 
            np.isnan(lips_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        jaw = jaw_4h_aligned[i]
        teeth = teeth_4h_aligned[i]
        lips = lips_4h_aligned[i]
        ema50_1d = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Alligator sleeping condition: jaw > teeth > lips OR lips > teeth > jaw (no clear trend)
        sleeping = ((jaw > teeth) and (teeth > lips)) or ((lips > teeth) and (teeth > jaw))
        # Awakening condition: teeth crosses jaw or lips (trend emerging)
        teeth_above_jaw = teeth > jaw
        teeth_above_lips = teeth > lips
        # Previous bar values for crossover detection
        if i > start_idx:
            prev_jaw = jaw_4h_aligned[i-1]
            prev_teeth = teeth_4h_aligned[i-1]
            prev_lips = lips_4h_aligned[i-1]
            prev_teeth_above_jaw = prev_teeth > prev_jaw
            prev_teeth_above_lips = prev_teeth > prev_lips
            # Teeth crosses jaw (bullish) or crosses lips (bearish)
            jaw_cross = (not prev_teeth_above_jaw) and teeth_above_jaw
            lips_cross = prev_teeth_above_lips and (not teeth_above_lips)
        else:
            jaw_cross = False
            lips_cross = False
        
        if position == 0:
            # Long: teeth crosses above jaw (bullish) AND uptrend (price > 1d EMA50) AND volume spike AND not sleeping
            long_condition = jaw_cross and (curr_close > ema50_1d) and volume_spike and (not sleeping)
            # Short: teeth crosses below lips (bearish) AND downtrend (price < 1d EMA50) AND volume spike AND not sleeping
            short_condition = lips_cross and (curr_close < ema50_1d) and volume_spike and (not sleeping)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or teeth crosses below lips (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or lips_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or teeth crosses above jaw (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or jaw_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0