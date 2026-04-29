#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Long when Alligator jaws (13) > teeth (8) > lips (5) AND price > 1d EMA34 AND volume > 1.5x 28-bar avg
# Short when Alligator jaws < teeth < lips AND price < 1d EMA34 AND volume > 1.5x 28-bar avg
# Exit when Alligator lines crossover (jaws crosses teeth) or opposite signal occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-25 trades/year on 12h.
# Williams Alligator identifies trending markets via smoothed moving averages. 1d EMA34 ensures
# alignment with higher timeframe trend, reducing counter-trend trades. Volume confirmation
# filters weak breakouts. Designed for low-frequency, high-conviction entries in both bull/bear.

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (SMMA-based)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Williams Alligator: three smoothed moving averages (SMMA)
    # Jaw (13-period), Teeth (8-period), Lips (5-period)
    def smma(arr, period):
        """Smoothed Moving Average - similar to EMA but with different smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA(i) = (SMMA(i-1)*(period-1) + arr[i]) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_12h, 13)   # Blue line
    teeth = smma(close_12h, 8)  # Red line
    lips = smma(close_12h, 5)   # Green line
    
    # Align Alligator lines to 12h timeframe (wait for 12h bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >1.5x 28-bar average volume (~14 periods on 12h = 1 week)
    volume_series = pd.Series(volume)
    volume_ma_28 = volume_series.rolling(window=28, min_periods=28).mean().values
    volume_confirm = volume > 1.5 * volume_ma_28
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 34)  # Volume MA and EMA need sufficient bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_28[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Alligator alignment: trending up when Jaw > Teeth > Lips
        alligator_long = jaw_val > teeth_val > lips_val
        # Alligator alignment: trending down when Jaw < Teeth < Lips
        alligator_short = jaw_val < teeth_val < lips_val
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Alligator trending up AND price > 1d EMA34 AND volume confirmation
            if alligator_long and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Alligator trending down AND price < 1d EMA34 AND volume confirmation
            elif alligator_short and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Alligator lines crossover (jaws crosses teeth) or opposite signal
            if jaw_val <= teeth_val or (jaw_val < teeth_val < lips_val and curr_close < ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Alligator lines crossover or opposite signal
            if jaw_val >= teeth_val or (jaw_val > teeth_val > lips_val and curr_close > ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals