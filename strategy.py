#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike Confirmation
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence/presence. 
When Alligator is "sleeping" (JAW > TEETH > LIPS for down, LIPS > TEETH > JAW for up) 
AND price breaks extreme (JAW for down, LIPS for up) with 1d EMA50 trend alignment 
and volume spike (>2.0x 20-bar vol MA), it captures strong trending moves. 
Works in bull via upside LIPS breaks and in bear via downside JAW breaks. 
12h timeframe targets 50-150 trades over 4 years. Discrete sizing (0.25) limits fee drag.
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
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h data (using prices directly as 12h timeframe)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Invalidate shifted values where shift creates look-ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (max shift 8), EMA50, and volume MA
    start_idx = max(51, 20, 13)  # 51 for EMA50 (50 + 1 for shift), 20 for volume MA, 13 for jaw
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        # Alligator sleep conditions (trending market)
        # Sleeping down: JAW > TEETH > LIPS (alligator mouth closed, bearish alignment)
        sleeping_down = (jaw_val > teeth_val) and (teeth_val > lips_val)
        # Sleeping up: LIPS > TEETH > JAW (alligator mouth closed, bullish alignment)
        sleeping_up = (lips_val > teeth_val) and (teeth_val > jaw_val)
        
        if position == 0:
            # Long: Alligator sleeping up + price breaks above LIPS + price above 1d EMA50 + volume confirmation
            long_signal = sleeping_up and (curr_high > lips_val) and price_above_ema and volume_confirm
            # Short: Alligator sleeping down + price breaks below JAW + price below 1d EMA50 + volume confirmation
            short_signal = sleeping_down and (curr_low < jaw_val) and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below LIPS OR price crosses below 1d EMA50
            if (curr_close < lips_val) or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above JAW OR price crosses above 1d EMA50
            if (curr_close > jaw_val) or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0