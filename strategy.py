#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume spike filter and chop regime avoidance
# Long when Alligator jaws (13) < teeth (8) < lips (5) AND 1d volume > 2.0x 20-period average AND chop > 61.8 (ranging)
# Short when Alligator jaws (13) > teeth (8) > lips (5) AND 1d volume > 2.0x 20-period average AND chop > 61.8 (ranging)
# Exit when Alligator lines cross (jaws-teeth or teeth-lips)
# Uses 4h primary timeframe with 1d HTF for volume confirmation and chop filter
# Williams Alligator catches trends early with smoothed moving averages
# Volume confirmation ensures breakouts have conviction
# Chop regime filter avoids whipsaws in strong trends
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_WilliamsAlligator_1dVolume_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume confirmation and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike filter
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_filter_1d = vol_1d > (2.0 * vol_ma_20)
    else:
        volume_filter_1d = np.zeros(len(df_1d), dtype=bool)
    
    # Calculate 1d chop filter (Choppiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    if len(high_1d) >= 14:
        atr_1d = []
        for i in range(1, len(high_1d)):
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
            atr_1d.append(tr)
        atr_1d = np.array(atr_1d)
        atr_sum_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        chop_denom = np.log10(max_high_14[13:] - min_low_14[13:]) * np.sqrt(14) * np.log10(100)
        chop_num = np.log10(atr_sum_14[13:]) * np.sqrt(14) * np.log10(100)
        chop_1d = np.full(len(high_1d), np.nan)
        chop_1d[13:] = 100 * chop_num / chop_denom
        chop_filter_1d = chop_1d > 61.8  # Chop > 61.8 = ranging market
    else:
        chop_filter_1d = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d filters to 4h timeframe
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    chop_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_filter_1d)
    
    # Get 4h data ONCE before loop for Williams Alligator
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Williams Alligator (Smoothed Moving Average)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    median_price_4h = (high_4h + low_4h) / 2  # Williams uses median price
    
    # Smoothed Moving Average (SMA with smoothing)
    def sma(arr, period):
        return pd.Series(arr).rolling(window=period, min_periods=period).mean().values
    
    def smma(arr, period):
        sma_vals = sma(arr, period)
        smma_vals = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            smma_vals[period-1] = sma_vals[period-1]
            for i in range(period, len(arr)):
                if not np.isnan(smma_vals[i-1]) and not np.isnan(arr[i]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + arr[i]) / period
        return smma_vals
    
    jaws = smma(median_price_4h, 13)  # Blue line (13-period)
    teeth = smma(median_price_4h, 8)   # Red line (8-period)
    lips = smma(median_price_4h, 5)    # Green line (5-period)
    
    # Align Alligator lines to 4h timeframe (same df_4h)
    jaws_aligned = align_htf_to_ltf(prices, df_4h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaws_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(volume_filter_1d_aligned[i]) or 
            np.isnan(chop_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator aligned up (jaws < teeth < lips) AND volume spike AND chop > 61.8 (ranging)
            if (jaws_aligned[i] < teeth_aligned[i] and 
                teeth_aligned[i] < lips_aligned[i] and 
                volume_filter_1d_aligned[i] and 
                chop_filter_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator aligned down (jaws > teeth > lips) AND volume spike AND chop > 61.8 (ranging)
            elif (jaws_aligned[i] > teeth_aligned[i] and 
                  teeth_aligned[i] > lips_aligned[i] and 
                  volume_filter_1d_aligned[i] and 
                  chop_filter_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines cross (jaws-teeth or teeth-lips)
            if (jaws_aligned[i] > teeth_aligned[i] or 
                teeth_aligned[i] > lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines cross (jaws-teeth or teeth-lips)
            if (jaws_aligned[i] < teeth_aligned[i] or 
                teeth_aligned[i] < lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals