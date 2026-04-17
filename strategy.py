#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + volume confirmation + ATR-based stoploss.
Long when price > Alligator Jaw AND Green > Red AND volume > 1.3x average.
Short when price < Alligator Jaw AND Green < Red AND volume > 1.3x average.
Exit when ATR trailing stop is hit (price < highest high since entry - 2.5*ATR for long,
price > lowest low since entry + 2.5*ATR for short).
Uses 4h for Alligator calculation and volume filter, 1d for chop regime to avoid whipsaw in ranging markets.
Target: 75-200 total trades over 4 years (19-50/year). Alligator identifies trend direction,
volume confirmation filters weak breakouts, ATR stop manages risk, chop filter avoids false signals in chop.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
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
    
    # Get 4h data for Williams Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Williams Alligator (Jaw=13, Teeth=8, Lips=5) with smoothing
    close_4h_series = pd.Series(close_4h)
    
    # Jaw (Blue line) - 13-period SMMA shifted 8 bars
    jaw_raw = close_4h_series.rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)  # shift 8 bars forward
    
    # Teeth (Red line) - 8-period SMMA shifted 5 bars
    teeth_raw = close_4h_series.rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)  # shift 5 bars forward
    
    # Lips (Green line) - 5-period SMMA shifted 3 bars
    lips_raw = close_4h_series.rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)  # shift 3 bars forward
    
    # Get 1d data for choppiness filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate choppiness index on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR (14-period)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = high_1d_series.rolling(window=14, min_periods=14).max().values
    ll = low_1d_series.rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr)/log(hh/ll)) / log10(14)
    sum_atr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    ratio = hh / ll
    ratio = np.where(ratio <= 1, 1.001, ratio)  # avoid division by zero or log<=0
    chop = 100 * (np.log10(sum_atr) - np.log10(ratio)) / np.log10(14)
    
    # Align indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips.values)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    # ATR for stoploss calculation (using 4h ATR)
    high_low = high_4h - low_4h
    high_close = np.abs(high_4h - np.roll(close_4h, 1))
    low_close = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_4h[0] = high_low[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        chop_val = chop_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        atr_val = atr_4h_aligned[i]
        
        if position == 0:
            # Long: price > Jaw AND Lips > Teeth (green above red) AND volume > 1.3x avg AND chop < 61.8 (trending)
            if price > jaw_val and lips_val > teeth_val and vol > 1.3 * vol_ma and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price < Jaw AND Lips < Teeth (green below red) AND volume > 1.3x avg AND chop < 61.8 (trending)
            elif price < jaw_val and lips_val < teeth_val and vol > 1.3 * vol_ma and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops below highest - 2.5*ATR
            if price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises above lowest + 2.5*ATR
            if price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0