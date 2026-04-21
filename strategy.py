#!/usr/bin/env python3
"""
6h_ChaikinOscillator_Energy_1dTrend_Confirmation
Hypothesis: Chaikin Oscillator (3,10) crossing zero combined with 1d trend filter (EMA50) and volume confirmation on 6b timeframe. Designed to capture momentum shifts with low trade frequency (~15-30/year) by requiring trend alignment and volume surge. Works in both bull/bear markets by following higher timeframe trend while using Chaikin for entry timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Chaikin Oscillator components on 6h data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)  # avoid division by zero
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # Accumulation/Distribution Line
    adl = np.cumsum(mfv)
    
    # Chaikin Oscillator: (3-period EMA of ADL) - (10-period EMA of ADL)
    adl_series = pd.Series(adl)
    ema_3_adl = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema_10_adl = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema_3_adl - ema_10_adl
    
    # === Volume confirmation: 20-period volume average on 6h ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    vol_ratio[np.isnan(vol_ratio)] = 1.0  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(chaikin_osc[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_1d = ema_50_1d_aligned[i]
        chaikin = chaikin_osc[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Chaikin crosses above zero + volume spike > 1.5 + price above 1d EMA50
            if chaikin > 0 and chaikin_osc[i-1] <= 0 and vol_spike > 1.5 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
            # Short: Chaikin crosses below zero + volume spike > 1.5 + price below 1d EMA50
            elif chaikin < 0 and chaikin_osc[i-1] >= 0 and vol_spike > 1.5 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when Chaikin crosses zero in opposite direction
            if position == 1 and chaikin < 0 and chaikin_osc[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and chaikin > 0 and chaikin_osc[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ChaikinOscillator_Energy_1dTrend_Confirmation"
timeframe = "6h"
leverage = 1.0