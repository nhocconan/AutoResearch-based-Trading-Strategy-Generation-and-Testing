#!/usr/bin/env python3
"""
6h_1d_Williams_Alligator_ADX_Filter
Hypothesis: Uses Williams Alligator (3 SMAs) from daily timeframe to determine trend,
with price above/below all three lines as long/short signal. ADX(14) from 1d filters
for trending markets (ADX > 25). Entry requires price to cross the Jaw line (13-period
SMMA) in direction of trend with volume confirmation. Designed to work in both bull
and bear markets by following daily trend while using 6h for precise entries. Targets
low trade frequency (12-37/year) via trend filter and entry conditions.
"""

name = "6h_1d_Williams_Alligator_ADX_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (used in Williams Alligator)"""
    s = pd.Series(series)
    return s.ewm(alpha=1/period, adjust=False).mean()

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Williams Alligator for Trend ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator lines: Jaw(13), Teeth(8), Lips(5) - all SMMA
    jaw = smma(df_1d['close'].values, 13)
    teeth = smma(df_1d['close'].values, 8)
    lips = smma(df_1d['close'].values, 5)
    
    # Align Alligator lines to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    
    # --- Daily ADX for Trend Strength Filter ---
    # Calculate ADX components
    plus_dm = np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0])
    minus_dm = np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0]))
    tr2 = np.abs(np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0]))
    tr3 = np.abs(np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smoothed values
    atr = smma(tr, 14)
    plus_di = 100 * smma(plus_dm, 14) / atr
    minus_di = 100 * smma(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smma(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- Volume Spike Detection (24-period average on 6h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or
            np.isnan(adx_6h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction based on Alligator alignment
        # Bullish: Lips > Teeth > Jaw (all aligned upward)
        bullish_alignment = lips_6h[i] > teeth_6h[i] and teeth_6h[i] > jaw_6h[i]
        # Bearish: Lips < Teeth < Jaw (all aligned downward)
        bearish_alignment = lips_6h[i] < teeth_6h[i] and teeth_6h[i] < jaw_6h[i]
        
        # Price position relative to Alligator
        price_above_all = close[i] > lips_6h[i] and close[i] > teeth_6h[i] and close[i] > jaw_6h[i]
        price_below_all = close[i] < lips_6h[i] and close[i] < teeth_6h[i] and close[i] < jaw_6h[i]
        
        # ADX trend filter
        trending = adx_6h[i] > 25
        
        # Volume confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: bullish alignment + price above Alligator + ADX > 25 + volume
            if bullish_alignment and price_above_all and trending and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + price below Alligator + ADX > 25 + volume
            elif bearish_alignment and price_below_all and trending and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or ADX weak
            if position == 1:
                # Exit long: bearish alignment OR ADX < 20
                if bearish_alignment or adx_6h[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish alignment OR ADX < 20
                if bullish_alignment or adx_6h[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals