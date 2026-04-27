#!/usr/bin/env python3
"""
12h_KAMA_Direction_Volume_Chop_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing reliable trend direction.
In trending markets (low chop), KAMA follows price closely; in ranging markets (high chop), it flattens.
We go long when price crosses above KAMA with volume confirmation in low-chop conditions,
and short when price crosses below KAMA with volume confirmation in low-chop conditions.
Uses 1d trend filter (EMA50) to ensure alignment with higher timeframe trend.
Designed for 12h timeframe to capture multi-day moves with low trade frequency.
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
    
    # KAMA parameters
    fast_ema = 2
    slow_ema = 30
    lookback = 10  # ER lookback period
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=lookback))
    abs_change = np.abs(np.diff(close, n=1))
    # Pad change array to match length
    change = np.concatenate([np.full(lookback, np.nan), change])
    # Sum of absolute changes over lookback period
    sum_abs_change = pd.Series(abs_change).rolling(window=lookback, min_periods=lookback).sum().values
    er = np.where(sum_abs_change > 0, change / sum_abs_change, 0)
    er = np.nan_to_num(er, nan=0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Choppiness Index filter (avoid choppy markets)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop_filter = chop < 61.8  # Only allow signals when not strongly ranging
    
    # Align indicators
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    volume_confirm_aligned = align_htf_to_ltf(prices, pd.DataFrame({'volume': volume}), volume_confirm)
    chop_filter_aligned = align_htf_to_ltf(prices, pd.DataFrame({'chop': chop}), chop_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need KAMA lookback (10), EMA50 (50), volume avg (20), chop (14)
    start_idx = max(lookback, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        chop_ok = chop_filter_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs 1d EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_conf and chop_ok:
                # Long when price crosses above KAMA
                if close_val > kama_val and (i == start_idx or close[i-1] <= kama_aligned[i-1]):
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf and chop_ok:
                # Short when price crosses below KAMA
                if close_val < kama_val and (i == start_idx or close[i-1] >= kama_aligned[i-1]):
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price crosses below KAMA or trend change
            if close_val < kama_val or close_val < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price crosses above KAMA or trend change
            if close_val > kama_val or close_val > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_Direction_Volume_Chop_Filter"
timeframe = "12h"
leverage = 1.0