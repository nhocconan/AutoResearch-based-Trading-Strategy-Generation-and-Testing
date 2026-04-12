#!/usr/bin/env python3
# 6h_1d_vix_fix_mean_reversion
# Hypothesis: Uses VIX Fix indicator on daily timeframe to detect volatility spikes in BTC/ETH
# that often precede mean-reverting moves. VIX Fix > 0.8 indicates fear, we look for reversals
# back to VWAP on 6h timeframe. Works in both bull/bear as volatility clustering persists.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.

name = "6h_1d_vix_fix_mean_reversion"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for VIX Fix calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate VIX Fix: measures market fear/volatility
    # VIX Fix = (Highest Close - Lowest Low) / ATR over lookback period
    highest_close = pd.Series(close_1d).rolling(window=22, min_periods=22).max().values
    atr_components = []
    atr_components.append(np.abs(high_1d - low_1d))
    atr_components.append(np.abs(high_1d - np.roll(close_1d, 1)))
    atr_components.append(np.abs(low_1d - np.roll(close_1d, 1)))
    true_range = np.maximum(np.maximum(atr_components[0], atr_components[1]), atr_components[2])
    atr = pd.Series(true_range).rolling(window=22, min_periods=22).mean().values
    
    # VIX Fix calculation
    vix_fix = (highest_close - low_1d) / atr
    # Invert so higher values = more fear (like VIX)
    vix_fix = np.where(vix_fix > 0, vix_fix, 0)
    
    # Align VIX Fix to 6h timeframe
    vix_fix_aligned = align_htf_to_ltf(prices, df_1d, vix_fix)
    
    # Calculate 6h VWAP for mean reversion target
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volumes) if 'volumes' in locals() else None
    # Fallback if volume not available in scope
    if 'volumes' not in locals():
        volumes = prices['volume'].values
    vwap_den = np.cumsum(volumes)
    vwap = np.where(vwap_den > 0, vwap_num / vwap_den, typical_price)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(vix_fix_aligned[i]) or np.isnan(vwap[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions: High fear (VIX Fix > 0.8) + price deviation from VWAP
        vix_threshold = 0.8
        price_dev = (close[i] - vwap[i]) / vwap[i]  # % deviation from VWAP
        
        # Long when fear is high and price is significantly below VWAP
        if (vix_fix_aligned[i] > vix_threshold and 
            price_dev < -0.015 and  # 1.5% below VWAP
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short when fear is high and price is significantly above VWAP
        elif (vix_fix_aligned[i] > vix_threshold and 
              price_dev > 0.015 and  # 1.5% above VWAP
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: fear subsides or price returns to VWAP
        elif position == 1 and (vix_fix_aligned[i] < 0.5 or price_dev > -0.005):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (vix_fix_aligned[i] < 0.5 or price_dev < 0.005):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals