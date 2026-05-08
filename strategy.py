#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Volume_Pressure_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d VWAP (Volume Weighted Average Price) ===
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_num = (typical_price_1d * df_1d['volume'].values)
    vwap_den = df_1d['volume'].values
    # Cumulative sum for VWAP
    cum_vwap_num = np.cumsum(vwap_num)
    cum_vwap_den = np.cumsum(vwap_den)
    vwap_1d = cum_vwap_num / (cum_vwap_den + 1e-10)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === 1d Volume Pressure: (Close - VWAP) / StdDev of typical price ===
    price_dev = typical_price_1d - vwap_1d
    # Rolling standard deviation of price deviation
    vol_pressure = price_dev / (pd.Series(price_dev).rolling(window=20, min_periods=20).std().values + 1e-10)
    vol_pressure_aligned = align_htf_to_ltf(prices, df_1d, vol_pressure)
    
    # === 6h Volume Spike: current volume > 1.5x 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 1.5)
    
    # === 6h Price position relative to VWAP ===
    price_vwap_ratio = close / (vwap_1d_aligned + 1e-10) - 1  # Normalized distance from VWAP
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_pressure_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for mean reversion when price deviates significantly from VWAP
            # with volume confirmation
            long_cond = (price_vwap_ratio[i] < -0.015 and  # Price >1.5% below VWAP
                        vol_pressure_aligned[i] < -0.8 and   # Strong selling pressure
                        vol_spike[i])                       # Volume spike
            
            short_cond = (price_vwap_ratio[i] > 0.015 and  # Price >1.5% above VWAP
                         vol_pressure_aligned[i] > 0.8 and   # Strong buying pressure
                         vol_spike[i])                      # Volume spike
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to VWAP or adverse pressure builds
            exit_cond = (price_vwap_ratio[i] > -0.005 or  # Back within 0.5% of VWAP
                        vol_pressure_aligned[i] > 0.3)     # Buying pressure weakening
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to VWAP or adverse pressure builds
            exit_cond = (price_vwap_ratio[i] < 0.005 or   # Back within 0.5% of VWAP
                        vol_pressure_aligned[i] < -0.3)    # Selling pressure weakening
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Mean reversion strategy based on 1d VWAP deviation with volume pressure
# confirmation. Enters when price deviates significantly (>1.5%) from daily VWAP
# accompanied by strong volume pressure and volume spikes. Exits when price returns
# to near VWAP or pressure dissipates. Works in both bull and bear markets as
# it fades extended moves regardless of trend direction. Targets 50-150 trades over
# 4 years to minimize fee drag. Uses discrete sizing (0.25).