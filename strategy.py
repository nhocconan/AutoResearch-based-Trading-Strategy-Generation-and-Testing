#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal strategy
# Uses 1d Camarilla levels for reversal signals with volume confirmation.
# Works in bull/bear markets as it fades extremes at R3/S3 levels.
# Target: 60-150 trades over 4 years (15-38/year) to balance frequency and cost.

name = "exp_12987_6h_camarilla_reversal_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    close_val = close
    h4 = close_val + range_val * 1.1 / 2
    l4 = close_val - range_val * 1.1 / 2
    h3 = close_val + range_val * 1.1 / 4
    l3 = close_val - range_val * 1.1 / 4
    h2 = close_val + range_val * 1.1 / 6
    l2 = close_val - range_val * 1.1 / 6
    h1 = close_val + range_val * 1.1 / 12
    l1 = close_val - range_val * 1.1 / 12
    return h1, h2, h3, h4, l1, l2, l3, l4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    h1_vals = np.zeros(len(close_d))
    h2_vals = np.zeros(len(close_d))
    h3_vals = np.zeros(len(close_d))
    h4_vals = np.zeros(len(close_d))
    l1_vals = np.zeros(len(close_d))
    l2_vals = np.zeros(len(close_d))
    l3_vals = np.zeros(len(close_d))
    l4_vals = np.zeros(len(close_d))
    
    for i in range(len(close_d)):
        h1, h2, h3, h4, l1, l2, l3, l4 = calculate_camarilla(high_d[i], low_d[i], close_d[i])
        h1_vals[i] = h1
        h2_vals[i] = h2
        h3_vals[i] = h3
        h4_vals[i] = h4
        l1_vals[i] = l1
        l2_vals[i] = l2
        l3_vals[i] = l3
        l4_vals[i] = l4
    
    # Align to 6h timeframe
    h1_aligned = align_htf_to_ltf(prices, df_daily, h1_vals)
    h2_aligned = align_htf_to_ltf(prices, df_daily, h2_vals)
    h3_aligned = align_htf_to_ltf(prices, df_daily, h3_vals)
    h4_aligned = align_htf_to_ltf(prices, df_daily, h4_vals)
    l1_aligned = align_htf_to_ltf(prices, df_daily, l1_vals)
    l2_aligned = align_htf_to_ltf(prices, df_daily, l2_vals)
    l3_aligned = align_htf_to_ltf(prices, df_daily, l3_vals)
    l4_aligned = align_htf_to_ltf(prices, df_daily, l4_vals)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Reversal at H3/L3 with volume
        reversal_long = volume_ok and close[i] <= l3_aligned[i]
        reversal_short = volume_ok and close[i] >= h3_aligned[i]
        
        # Generate signals
        if position == 0:
            if reversal_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif reversal_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals