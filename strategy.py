#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_MA_Combo
Hypothesis: Combine Elder Ray (Bull/Bear Power) with ZeroLag MA on 6h timeframe, filtered by 12h trend (EMA50). 
Elder Ray identifies institutional buying/selling pressure, ZeroLag MA reduces lag for timely entries, 
and 12h EMA50 ensures we trade with higher timeframe trend. Discrete sizing 0.25 targets ~12-37 trades/year.
Works in bull/bear via 12h trend filter and avoids whipsaws via Elder Ray confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA13 and EMA21 for ZeroLag MA (6h)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # ZeroLag MA: 2*EMA - EMA(EMA)
    zl_ema13 = 2 * ema13 - pd.Series(ema13).ewm(span=13, adjust=False, min_periods=13).mean().values
    zl_ema21 = 2 * ema21 - pd.Series(ema21).ewm(span=21, adjust=False, min_periods=21).mean().values
    zlma = (zl_ema13 + zl_ema21) / 2
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_for_elder = ema13  # reuse EMA13
    bull_power = high - ema13_for_elder
    bear_power = low - ema13_for_elder
    
    # Smooth Elder Ray with 5-period EMA to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 21 for EMA21, 5 for smoothing
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(zlma[i]) or
            np.isnan(bull_power_smooth[i]) or
            np.isnan(bear_power_smooth[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        zlma_val = zlma[i]
        bull_val = bull_power_smooth[i]
        bear_val = bear_power_smooth[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > ZLMA, Bull Power > 0, Bear Power < 0, and 12h uptrend (close > EMA50)
            long_entry = (close_val > zlma_val) and (bull_val > 0) and (bear_val < 0) and (close_val > ema_50_12h_val)
            # Short: price < ZLMA, Bull Power < 0, Bear Power > 0, and 12h downtrend (close < EMA50)
            short_entry = (close_val < zlma_val) and (bull_val < 0) and (bear_val > 0) and (close_val < ema_50_12h_val)
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or Elder Ray divergence
            if (close_val < ema_50_12h_val) or (bull_val < 0) or (bear_val > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or Elder Ray divergence
            if (close_val > ema_50_12h_val) or (bull_val > 0) or (bear_val < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_ZeroLag_MA_Combo"
timeframe = "6h"
leverage = 1.0