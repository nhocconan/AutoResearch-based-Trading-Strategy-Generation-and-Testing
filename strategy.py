#!/usr/bin/env python3
# 6h_12h_orderflow_imbalance_v1
# Hypothesis: 6-hour price reversals driven by order flow imbalances at 12-hour pivot zones.
# Uses 12-hour volume-weighted average price (VWAP) and standard deviation bands to detect
# overextended moves. Long when price crosses below VWAP - 1*SD with bullish order flow
# imbalance (buying pressure), short when price crosses above VWAP + 1*SD with bearish
# imbalance. Works in both bull/bear markets as VWAP adapts to price action and
# order flow filters reduce whipsaw. Target: 60-120 total trades over 4 years (15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_orderflow_imbalance_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    
    # Load 12h data ONCE before loop for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate VWAP and standard deviation for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    tp_vol_12h = typical_price_12h * volume_12h
    
    # Cumulative VWAP components (reset each 12h bar)
    cum_tp_vol = np.zeros(len(df_12h))
    cum_vol = np.zeros(len(df_12h))
    vwap_12h = np.full(len(df_12h), np.nan)
    
    for i in range(len(df_12h)):
        cum_tp_vol[i] = np.sum(tp_vol_12h[:i+1])
        cum_vol[i] = np.sum(volume_12h[:i+1])
        if cum_vol[i] > 0:
            vwap_12h[i] = cum_tp_vol[i] / cum_vol[i]
    
    # Calculate VWAP standard deviation
    vwap_std_12h = np.full(len(df_12h), np.nan)
    for i in range(len(df_12h)):
        if cum_vol[i] > 0:
            # Weighted variance calculation
            squared_diff = (typical_price_12h[:i+1] - vwap_12h[i]) ** 2
            weighted_squared_diff = squared_diff * volume_12h[:i+1]
            variance = np.sum(weighted_squared_diff) / cum_vol[i]
            vwap_std_12h[i] = np.sqrt(variance) if variance > 0 else 0.0
    
    # Define bands: VWAP ± 1 standard deviation
    upper_band_12h = vwap_12h + vwap_std_12h
    lower_band_12h = vwap_12h - vwap_std_12h
    
    # Align 12h VWAP and bands to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band_12h)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band_12h)
    
    # Order flow imbalance: buying/selling pressure
    # Calculate money flow index components for 6h data
    money_flow = tp_vol  # typical price * volume
    # Positive and negative money flow
    pmf = np.where(close > np.roll(close, 1), money_flow, 0)
    nmf = np.where(close < np.roll(close, 1), money_flow, 0)
    # Handle first element
    pmf[0] = 0
    nmf[0] = 0
    
    # Calculate 12-period money flow ratio for imbalance
    mfr = np.full(n, np.nan)
    for i in range(11, n):
        pos_mf = np.sum(pmf[i-10:i+1])
        neg_mf = np.sum(nmf[i-10:i+1])
        if neg_mf != 0:
            mfr[i] = pos_mf / neg_mf
        else:
            mfr[i] = 100.0  # All positive flow
    
    # Define imbalance thresholds
    bullish_imbalance = mfr > 1.8  # Strong buying pressure
    bearish_imbalance = mfr < 0.6  # Strong selling pressure
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(vwap_aligned[i]) or np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or above VWAP
            if close[i] >= vwap_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or below VWAP
            if close[i] <= vwap_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price crosses below lower band with bullish order flow imbalance
            if close[i] < lower_band_aligned[i] and bullish_imbalance[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses above upper band with bearish order flow imbalance
            elif close[i] > upper_band_aligned[i] and bearish_imbalance[i]:
                position = -1
                signals[i] = -0.25
    
    return signals