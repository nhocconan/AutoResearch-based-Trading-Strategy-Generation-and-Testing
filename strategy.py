#!/usr/bin/env python3
"""
Hypothesis: 1-day Parabolic SAR trend following with weekly Supertrend filter and volume confirmation.
Long when price > PSAR, weekly Supertrend is bullish, and volume > 1.5x average.
Short when price < PSAR, weekly Supertrend is bearish, and volume > 1.5x average.
Exit when price crosses PSAR or weekly Supertrend flips.
Designed for low trade frequency (~10-25/year) to capture major trends while avoiding whipsaws.
Works in both bull and bear markets by requiring trend alignment across timeframes.
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
    
    # Load daily data for PSAR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load weekly data for Supertrend - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Parabolic SAR (0.02 step, 0.2 max)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    psar = np.zeros_like(close_1d)
    psar[0] = low_1d[0]
    
    # Initialize
    bull = True
    af = 0.02
    ep = high_1d[0] if bull else low_1d[0]
    
    for i in range(1, len(close_1d)):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR doesn't exceed prior lows
            psar[i] = min(psar[i], low_1d[i-1], low_1d[i-2] if i >= 2 else low_1d[i-1])
            if low_1d[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = 0.02
                ep = low_1d[i]
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR doesn't fall below prior highs
            psar[i] = max(psar[i], high_1d[i-1], high_1d[i-2] if i >= 2 else high_1d[i-1])
            if high_1d[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = 0.02
                ep = high_1d[i]
        
        # Update acceleration factor and extreme point
        if bull:
            if high_1d[i] > ep:
                ep = high_1d[i]
                af = min(af + 0.02, 0.2)
        else:
            if low_1d[i] < ep:
                ep = low_1d[i]
                af = min(af + 0.02, 0.2)
    
    # Calculate weekly Supertrend (10, 3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR
    atr_period = 10
    atr = np.zeros_like(close_1w)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1w + low_1w) / 2 + 3.0 * atr
    basic_lb = (high_1w + low_1w) / 2 - 3.0 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close_1w)
    final_lb = np.zeros_like(close_1w)
    supertrend = np.zeros_like(close_1w, dtype=bool)  # True = uptrend
    
    for i in range(len(close_1w)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]) else final_ub[i-1]
            final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]) else final_lb[i-1]
        
        if i == 0:
            supertrend[i] = True
        else:
            if supertrend[i-1]:
                supertrend[i] = close_1w[i] > final_ub[i-1]
            else:
                supertrend[i] = close_1w[i] < final_lb[i-1]
    
    # Align HTF indicators to lower timeframe
    psar_aligned = align_htf_to_ltf(prices, df_1d, psar)
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend.astype(float))
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(psar_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        psar_val = psar_aligned[i]
        supertrend_val = supertrend_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: price > PSAR, Supertrend bullish, volume confirmation
            if (close_val > psar_val and supertrend_val > 0.5 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price < PSAR, Supertrend bearish, volume confirmation
            elif (close_val < psar_val and supertrend_val < 0.5 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price < PSAR OR Supertrend bearish
                if close_val < psar_val or supertrend_val < 0.5:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price > PSAR OR Supertrend bullish
                if close_val > psar_val or supertrend_val > 0.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_PSAR_1wSupertrend_Volume"
timeframe = "1d"
leverage = 1.0