#!/usr/bin/env python3
"""
12h Williams Alligator with 1d/1w EMA Trend Filter and Volume Spike
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend absence/presence. 
When aligned with 1d/1w EMA50 trend and volume confirmation, it filters false signals.
Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years) by requiring
confluence of Alligator alignment, multi-timeframe trend, and volume spike.
Works in bull (long when Alligator bullish + uptrend) and bear (short when Alligator bearish + downtrend).
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 12h (primary timeframe)
    # JAW (Blue): 13-period SMMA, shifted 8 bars
    # TEETH (Red): 8-period SMMA, shifted 5 bars  
    # LIPS (Green): 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1w EMA50 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current volume > 1.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (13+8 shift), EMA50, and volume
    start_idx = max(13 + 8, 50)  # Alligator jaw shift, EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: check if lines are properly ordered
        # Bullish: Lips > Teeth > Jaw (alligator mouth opening up)
        # Bearish: Lips < Teeth < Jaw (alligator mouth opening down)
        bullish_alligator = (lips_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > jaw_shifted[i])
        bearish_alligator = (lips_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < jaw_shifted[i])
        
        # Trend filter: price relative to HTF EMAs
        bullish_trend = (curr_close > ema_1d_aligned[i]) and (curr_close > ema_1w_aligned[i])
        bearish_trend = (curr_close < ema_1d_aligned[i]) and (curr_close < ema_1w_aligned[i])
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + trend + volume
            # Long: Alligator bullish AND bullish trend AND volume spike
            long_entry = bullish_alligator and bullish_trend and vol_spike
            # Short: Alligator bearish AND bearish trend AND volume spike
            short_entry = bearish_alligator and bearish_trend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator turns bearish OR loss of bullish trend
            if not bullish_alligator or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator turns bullish OR loss of bearish trend
            if not bearish_alligator or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1d1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0