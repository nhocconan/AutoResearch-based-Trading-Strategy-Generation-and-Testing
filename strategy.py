#!/usr/bin/env python3
"""
Hypothesis: 12h EMA crossover (20/50) + volume confirmation + daily volatility filter.
Uses EMA crossover for trend direction, volume > 20-period average for confirmation,
and daily ATR below median to avoid high-volatility chop.
Works in both bull/bear by following EMA trend. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ema_crossover_vol_volatility_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === EMA CROSSOVER (LTF) ===
    ema_fast = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_slow = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === DAILY VOLATILITY FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # True Range
    tr1 = d_high - d_low
    tr2 = np.abs(d_high - np.roll(d_close, 1))
    tr3 = np.abs(d_low - np.roll(d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily ATR median (for volatility filter)
    atr_median = np.full_like(atr, np.nan)
    for i in range(len(atr)):
        if i >= 30:
            atr_median[i] = np.nanmedian(atr[max(0, i-30):i])
    
    # Align ATR median to 12h timeframe
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median)
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_median_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is below median (low volatility)
        if atr[i] > atr_median_aligned[i]:
            signals[i] = 0.0
            continue
        
        # EMA crossover signals
        bullish_cross = ema_fast[i] > ema_slow[i]
        bearish_cross = ema_fast[i] < ema_slow[i]
        
        if position == 1:  # Long position
            # Exit: bearish crossover OR volume drops below average
            if bearish_cross or volume[i] <= vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish crossover OR volume drops below average
            if bullish_cross or volume[i] <= vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry on EMA crossover with volume confirmation
            if bullish_cross:
                position = 1
                signals[i] = 0.25
            elif bearish_cross:
                position = -1
                signals[i] = -0.25
    
    return signals