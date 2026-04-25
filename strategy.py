#!/usr/bin/env python3
"""
4h Williams Alligator Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (all lines intertwined) vs presence (lines diverged). 
Breakout above Lips in bullish Alligator alignment (Lips>Teeth>Jaw) or below Jaw in bearish alignment (Jaw>Teeth>Lips) 
with volume confirmation captures trending moves. EMA34 trend filter ensures we trade with higher timeframe momentum. 
Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend). 
Target 20-30 trades/year on 4h to avoid fee drag.
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
    
    # Get 1d data for Williams Alligator and EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods  
    # Lips: 5-period SMMA smoothed by 3 periods
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    median_price_1d = (high_1d + low_1d) / 2.0  # Typical price
    
    jaw_raw = smma(median_price_1d, 13)
    teeth_raw = smma(median_price_1d, 8)
    lips_raw = smma(median_price_1d, 5)
    
    jaw = smma(jaw_raw, 8)
    teeth = smma(teeth_raw, 5)
    lips = smma(lips_raw, 3)
    
    # Align Alligator lines to 4h timeframe
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
    
    # Start index: need enough for Alligator, EMA34, ATR, volume MA
    start_idx = max(34, 14, 20, 35)  # 35 for Alligator smoothing
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
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
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Alligator alignment: bullish (Lips > Teeth > Jaw) or bearish (Jaw > Teeth > Lips)
        bullish_alligator = lips_val > teeth_val > jaw_val
        bearish_alligator = jaw_val > teeth_val > lips_val
        
        # Trend filter: price relative to 1d EMA34 (require 0.3% deviation to avoid whipsaw)
        uptrend = curr_close > ema_34_val * 1.003
        downtrend = curr_close < ema_34_val * 0.997
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Lips with bullish Alligator and volume confirmation in uptrend
            long_breakout = (curr_close > lips_val) and bullish_alligator and volume_confirm and uptrend
            # Short: price breaks below Jaw with bearish Alligator and volume confirmation in downtrend
            short_breakout = (curr_close < jaw_val) and bearish_alligator and volume_confirm and downtrend
            
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
            # Exit conditions: price closes below Jaw OR 2.0*ATR trailing stop OR Alligator turns bearish
            if curr_close < jaw_val or curr_close < (highest_since_entry - 2.0 * atr_val) or not bullish_alligator:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above Lips OR 2.0*ATR trailing stop OR Alligator turns bullish
            if curr_close > lips_val or curr_close > (lowest_since_entry + 2.0 * atr_val) or bullish_alligator:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0