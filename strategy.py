#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when Jaw > Teeth > Lips (bearish alignment) AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when Alligator alignment breaks (jaws cross teeth or lips)
# Uses discrete position sizing (0.25) to reduce fee drag.
# Target: 15-40 trades/year on 4h timeframe (60-160 total over 4 years) to avoid overtrading.
# Williams Alligator identifies trend phases via smoothed medians, effective in both bull and bear markets.

name = "4h_WilliamsAlligator_1dEMA34_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator components (SMMA with 5-period, then shifted)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price_1d = (high_1d + low_1d) / 2
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to RMA/Wilder's smoothing
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Value) / Period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw_1d = smma(median_price_1d, 13)
    teeth_1d = smma(median_price_1d, 8)
    lips_1d = smma(median_price_1d, 5)
    
    # Shift the Alligator lines (Jaw: 8, Teeth: 5, Lips: 3)
    jaw_1d_shifted = np.roll(jaw_1d, 8)
    teeth_1d_shifted = np.roll(teeth_1d, 5)
    lips_1d_shifted = np.roll(lips_1d, 3)
    
    # Set NaN for shifted periods
    jaw_1d_shifted[:8] = np.nan
    teeth_1d_shifted[:5] = np.nan
    lips_1d_shifted[:3] = np.nan
    
    # Align Alligator components to 4h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d_shifted)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d_shifted)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d_shifted)
    
    # Volume confirmation: >2.0x 20-bar average volume (tight to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 13)  # volume MA, EMA34, and Alligator warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_1d_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_jaw = jaw_1d_aligned[i]
        curr_teeth = teeth_1d_aligned[i]
        curr_lips = lips_1d_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks (jaw crosses above teeth or lips)
            if curr_jaw > curr_teeth or curr_jaw > curr_lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (jaw crosses below teeth or lips)
            if curr_jaw < curr_teeth or curr_jaw < curr_lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Bullish alignment: Jaw < Teeth < Lips
            bullish_alignment = (curr_jaw < curr_teeth) and (curr_teeth < curr_lips)
            # Bearish alignment: Jaw > Teeth > Lips
            bearish_alignment = (curr_jaw > curr_teeth) and (curr_teeth > curr_lips)
            
            # Long when bullish alignment AND price > 1d EMA34 AND volume confirmation
            if bullish_alignment and curr_close > curr_ema34_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment AND price < 1d EMA34 AND volume confirmation
            elif bearish_alignment and curr_close < curr_ema34_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals