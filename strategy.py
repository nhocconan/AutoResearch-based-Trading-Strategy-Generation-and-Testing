#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter, volume spike (>2.0x 20-bar avg), and choppiness regime (CHOP < 38.2 = trend). Uses discrete 0.25 position sizing. Alligator provides trend direction, volume confirms momentum, chop filter avoids ranging markets. Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag in 6h timeframe.

name = "6h_WilliamsAlligator_1dEMA34_VolumeChopRegime_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts: Jaw shifted 8 bars, Teeth shifted 5 bars, Lips shifted 3 bars
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align to LTF
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted) if len(df_1d) > 0 else np.full(n, np.nan)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted) if len(df_1d) > 0 else np.full(n, np.nan)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted) if len(df_1d) > 0 else np.full(n, np.nan)
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Choppiness Index (CHOP) on 14-period for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / true_range_sum) / np.log10(14)
    chop = np.where(true_range_sum == 0, 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish Alligator alignment, price above 1d EMA34, volume spike (>2.0x), trending regime (CHOP < 38.2)
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i] and 
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment, price below 1d EMA34, volume spike (>2.0x), trending regime (CHOP < 38.2)
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i] and 
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks OR chop becomes too high (choppy market)
            if (lips_aligned[i] < teeth_aligned[i] or 
                teeth_aligned[i] < jaw_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks OR chop becomes too high (choppy market)
            if (lips_aligned[i] > teeth_aligned[i] or 
                teeth_aligned[i] > jaw_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals