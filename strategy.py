#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with 1w EMA34 trend filter, volume confirmation (>1.5x 20-bar avg volume), and weekly choppiness regime filter (CHOP > 61.8 = range -> avoid entries, CHOP < 38.2 = trend -> allow breakout entries). Uses 6h timeframe to target 50-150 total trades over 4 years. Williams Alligator (Jaw/Teeth/Lips) provides trend direction and dynamic support/resistance. Weekly trend filter and regime filter reduce false signals in bear markets. Discrete position sizing (0.25) minimizes fee churn.

name = "6h_WilliamsAlligator_1wEMA34_VolumeChopRegime_v1"
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
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Choppiness Index (CHOP) on 14-period for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w_arr, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w_arr, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop_1w = 100 * np.log10(atr_sum / true_range_sum) / np.log10(14)
    chop_1w = np.where(true_range_sum == 0, 50, chop_1w)  # avoid div by zero
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
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
    # Set shifted values to NaN for invalid positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align to LTF (already accounts for completed bar timing)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_shifted) if len(df_1w) > 0 else np.full(n, np.nan)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_shifted) if len(df_1w) > 0 else np.full(n, np.nan)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_shifted) if len(df_1w) > 0 else np.full(n, np.nan)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment), close > 1w EMA34, volume spike (>1.5x avg), trending regime (CHOP < 38.2)
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                chop_1w_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment), close < 1w EMA34, volume spike (>1.5x avg), trending regime (CHOP < 38.2)
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  chop_1w_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if Alligator alignment breaks (Lips < Teeth OR Teeth < Jaw) OR chop becomes too high (choppy market)
            if (lips_aligned[i] < teeth_aligned[i] or 
                teeth_aligned[i] < jaw_aligned[i] or 
                chop_1w_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if Alligator alignment breaks (Lips > Teeth OR Teeth > Jaw) OR chop becomes too high (choppy market)
            if (lips_aligned[i] > teeth_aligned[i] or 
                teeth_aligned[i] > jaw_aligned[i] or 
                chop_1w_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals