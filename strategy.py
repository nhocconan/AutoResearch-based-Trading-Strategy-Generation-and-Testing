#!/usr/bin/env python3
"""
6h Elder Ray Power with Volume Confirmation and ATR Filter
Hypothesis: Elder Ray Bull/Bear Power (EMA13 minus High/Low) indicates institutional buying/selling pressure.
Combined with volume confirmation (>1.5x average) and ATR filter (>0.5% of price) to avoid chop.
Works in bull markets (buy power > 0) and bear markets (bear power < 0) by fading extremes.
Target: 15-30 trades/year to minimize fee drain.
"""

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
    
    # Get 1d data for Elder Ray calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Elder Ray components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 of close for 1d
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Align to 6h timeframe with proper delay (use previous day's values)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators (max of 13,20,14)
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        atr_val = atr[i]
        vol_conf = vol_ratio[i] > 1.5
        
        # Volatility filter: avoid choppy markets (ATR > 0.5% of price)
        vol_filter = atr_val > 0.005 * price
        
        if position == 0:
            # Strong bull power with volume and volatility confirmation = long
            if bull > 0 and vol_conf and vol_filter:
                signals[i] = 0.25
                position = 1
            # Strong bear power with volume and volatility confirmation = short
            elif bear > 0 and vol_conf and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if bull power turns negative or volatility drops
            if bull <= 0 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if bear power turns negative or volatility drops
            if bear <= 0 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Elder_Ray_Power_Volume_ATR"
timeframe = "6h"
leverage = 1.0