#!/usr/bin/env python3
"""
12h Williams Alligator Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence/presence. 
Breakouts above Alligator lips in uptrend or below jaws in downtrend with volume confirmation 
capture strong directional moves. Works in bull (buy Alligator breaks in uptrend) and bear 
(sell Alligator breaks in downtrend) via symmetric logic. Target 12-30 trades/year on 12h.
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
    
    # Get 1d data for EMA50 trend filter and Alligator components (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator components on 1d
    # Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_1d_series = df_1d['close'].values
    jaw_raw = smma(close_1d_series, 13)
    teeth_raw = smma(close_1d_series, 8)
    lips_raw = smma(close_1d_series, 5)
    
    # Shift jaw forward by 8, teeth by 5, lips by 3 (Alligator alignment)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Align Alligator components to 12h timeframe (no extra delay for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate ATR(14) for stop management
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA50, ATR, volume MA, Alligator
    start_idx = max(50, 14, 20, 13)  # EMA50 needs 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Alligator conditions: Lips above Teeth above Jaw = uptrend, reverse = downtrend
        alligator_bullish = lips_val > teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Lips with volume confirmation in uptrend and Alligator bullish
            long_breakout = (curr_close > lips_val) and volume_confirm and uptrend and alligator_bullish
            # Short: price breaks below Jaw with volume confirmation in downtrend and Alligator bearish
            short_breakout = (curr_close < jaw_val) and volume_confirm and downtrend and alligator_bearish
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below Jaw OR 2.5*ATR trailing stop OR EMA50 trend turns down
            if curr_close < jaw_val or curr_close < (highest_since_entry - 2.5 * atr_val) or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above Lips OR 2.5*ATR trailing stop OR EMA50 trend turns up
            if curr_close > lips_val or curr_close > (lowest_since_entry + 2.5 * atr_val) or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0