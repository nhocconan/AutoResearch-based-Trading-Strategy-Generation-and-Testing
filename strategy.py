#!/usr/bin/env python3
"""
6h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V2
Hypothesis: Use daily Camarilla pivot levels (R1/S1) for breakout entries on 6h timeframe.
Long when price breaks above R1 with volume > 1.5x average and ATR filter (low volatility).
Short when price breaks below S1 with same conditions.
Exit when price returns to the pivot point (PP) or reverses at opposite level (S1 for longs, R1 for shorts).
Uses ATR filter to avoid choppy markets. Works in bull via breakouts, bear via reversals at S1/R1.
Target: 20-40 trades/year by requiring multiple confirmations (level break + volume + volatility filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # R2 = close + 0.5 * (high - low)
    # R1 = close + 0.25 * (high - low)
    # PP = (high + low + close) / 3
    # S1 = close - 0.25 * (high - low)
    # S2 = close - 0.5 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    
    rng = high_1d - low_1d
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + 0.25 * rng
    s1 = close_1d - 0.25 * rng
    
    # Align daily pivot levels to 6h timeframe (using previous day's close for calculation)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # ATR filter (6-period ATR on 6s timeframe)
    atr_period = 6
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first period
    atr = np.full_like(close, np.nan)
    
    if len(tr) >= atr_period:
        for i in range(atr_period, len(tr)):
            atr[i] = np.mean(tr[i - atr_period:i])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Session filter: 08-20 UTC (avoid low liquidity)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # ATR filter: only trade when volatility is not too high (avoid chop)
        # Use ATR as percentage of price - only trade when ATR% < 0.03 (3%)
        atr_percent = atr[i] / close[i] if close[i] > 0 else 0
        vol_filter = atr_percent < 0.03
        
        if position == 0 and in_session:
            # Long: price breaks above R1 + volume + volatility filter
            if close[i] > r1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + volatility filter
            elif close[i] < s1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to PP or breaks below S1 (reversal)
            if close[i] <= pp_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to PP or breaks above R1 (reversal)
            if close[i] >= pp_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V2"
timeframe = "6h"
leverage = 1.0