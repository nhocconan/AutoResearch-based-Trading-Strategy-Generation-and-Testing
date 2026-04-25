#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) on 12h defines trend direction,
1d EMA34 filters higher-timeframe trend alignment, volume spike confirms momentum,
and chop regime filter avoids ranging markets. Designed for 12h timeframe to
capture multi-day swings in both bull and bear markets with low trade frequency.
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
    
    # Get 12h data for Williams Alligator (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h: SMAs with offsets
    # JAW: 13-period SMMA shifted 8 bars
    # TEETH: 8-period SMMA shifted 5 bars  
    # LIPS: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_12h = smma(df_12h['close'].values, 13)
    teeth_12h = smma(df_12h['close'].values, 8)
    lips_12h = smma(df_12h['close'].values, 5)
    
    # Shift as per Alligator definition
    jaw_12h = np.roll(jaw_12h, 8)
    teeth_12h = np.roll(teeth_12h, 5)
    lips_12h = np.roll(lips_12h, 3)
    # Set shifted values to NaN
    jaw_12h[:8] = np.nan
    teeth_12h[:5] = np.nan
    lips_12h[:3] = np.nan
    
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Calculate Choppiness Index (CHOP) for regime filter
    if len(close) >= 14:
        atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
        hh = pd.Series(high).rolling(window=14, min_periods=14).max()
        ll = pd.Series(low).rolling(window=14, min_periods=14).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
        chop_values = chop.values
    else:
        chop_values = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Alligator, EMA34, ATR, and volume MA
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        jaw = jaw_12h_aligned[i]
        teeth = teeth_12h_aligned[i]
        lips = lips_12h_aligned[i]
        ema34_1d = ema_34_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        chop = chop_values[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips > teeth and teeth > jaw
        alligator_short = lips < teeth and teeth < jaw
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        # Chop filter: only trade when trending (CHOP < 38.2)
        trending_regime = chop < 38.2
        
        if position == 0:
            # Long: Alligator aligned up AND price > 1d EMA34 AND volume spike AND trending regime
            long_condition = alligator_long and (curr_close > ema34_1d) and volume_spike and trending_regime
            # Short: Alligator aligned down AND price < 1d EMA34 AND volume spike AND trending regime
            short_condition = alligator_short and (curr_close < ema34_1d) and volume_spike and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or Alligator reverses (Lips < Jaw)
            if curr_close <= entry_price - 2.5 * atr_val or lips < jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or Alligator reverses (Lips > Jaw)
            if curr_close >= entry_price + 2.5 * atr_val or lips > jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0