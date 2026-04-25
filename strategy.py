#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on 6h identifies trend alignment, 
while 1d EMA50 filters for higher-timeframe trend direction. Volume spike confirms 
momentum. Works in bull markets via long alignments and in bear via short alignments.
Uses ATR-based trailing stop. Targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Alligator and 1d data for EMA50 (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_6h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA*(period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(df_6h['close'].values, 13)
    teeth_raw = smma(df_6h['close'].values, 8)
    lips_raw = smma(df_6h['close'].values, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Align Alligator components to 6h timeframe (already aligned via get_htf_data)
    jaw_aligned = jaw  # Already on 6h bars
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss (6h)
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for Alligator, EMA50_1d, volume MA, ATR to propagate
    start_idx = max(50, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        ema50_1d = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_align = (lips > teeth) and (teeth > jaw)
        bearish_align = (lips < teeth) and (teeth < jaw)
        
        if position == 0:
            # Long: Alligator bullish + price above 1d EMA50 + volume confirmation
            long_condition = bullish_align and (curr_close > ema50_1d) and volume_confirm
            # Short: Alligator bearish + price below 1d EMA50 + volume confirmation
            short_condition = bearish_align and (curr_close < ema50_1d) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 2.0 * atr  # Initial stop
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 2.0 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 2.0*ATR
            atr_stop = max(atr_stop, curr_high - 2.0 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 2.0*ATR
            atr_stop = min(atr_stop, curr_low + 2.0 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0