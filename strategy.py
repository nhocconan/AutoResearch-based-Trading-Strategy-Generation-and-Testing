#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
Alligator lines (Jaw, Teeth, Lips) indicate trend strength and direction.
In bull market (1d close > 1d EMA200): long when Lips > Teeth > Jaw (bullish alignment).
In bear market (1d close < 1d EMA200): short when Lips < Teeth < Jaw (bearish alignment).
Volume must be above 20-period average to confirm trend strength.
This combines trend following with trend filter and volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_alligator_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)  # already shifted
    
    # === WILLIAMS ALLIGATOR (LTF) ===
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        # Smoothed Moving Average (similar to Wilder's smoothing)
        sma = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            result[period-1] = sma[period-1]
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (Jaw +8, Teeth +5, Lips +3)
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after sufficient data for SMMA and shifts
        if (np.isnan(one_d_ema_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        bull_trend = close[i] > one_d_ema_aligned[i]
        
        # Alligator alignment
        bullish_alignment = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
        bearish_alignment = lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i]
        
        if position == 1:  # Long position
            # Exit: bearish alignment OR trend turns bearish
            if not bullish_alignment or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish alignment OR trend turns bullish
            if not bearish_alignment or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on 1d trend and Alligator alignment
            if bull_trend:
                # In bull market: long on bullish Alligator alignment
                if bullish_alignment:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on bearish Alligator alignment
                if bearish_alignment:
                    position = -1
                    signals[i] = -0.25
    
    return signals