#!/usr/bin/env python3
"""
12h Williams Alligator Breakout with 1d EMA34 Trend Filter and Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence. 
When price breaks above/below the Alligator's mouth with volume confirmation and 
aligned with 1d EMA34 trend, it captures strong momentum moves. ATR trailing stop 
manages risk. Designed for 12h timeframe to avoid overtrading (target: 12-37 trades/year).
Works in bull/bear markets by following the 1d EMA trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line  
    lips = smma(close, 5)   # Green line
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # ATR for volatility filter and trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA34 trend filter (MTF) - loaded ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators (Alligator needs max period + shifts)
    start_idx = max(13, 8, 5) + 8 + 30 + 1  # jaw period + jaw shift + vol lookback + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_14[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator mouth: outsider lines (lips and teeth)
        # When lips > teeth > jaw: bullish alignment (mouth open up)
        # When lips < teeth < jaw: bearish alignment (mouth open down)
        bullish_aligned = lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]
        bearish_aligned = lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]
        
        # Breakout conditions: price breaks above/below Alligator's mouth
        breakout_long = curr_close > lips_shifted[i] and bullish_aligned
        breakout_short = curr_close < jaw_shifted[i] and bearish_aligned
        
        if position == 0:
            # Look for entry signals - require: Alligator breakout + volume spike + 1d EMA trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_34_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: ATR trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.5 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0