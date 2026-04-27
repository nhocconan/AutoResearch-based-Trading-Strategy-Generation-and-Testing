#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) with 1d Bollinger Band width regime filter.
# In expanding volatility regimes (BB width rising), price tends to revert to VWAP.
# In contracting regimes (BB width falling), price trends away from VWAP.
# Long when price < VWAP and BB width expanding; short when price > VWAP and BB width expanding.
# Flat when BB width contracting (trending regime). Uses volume confirmation to avoid false signals.
# Works in both bull (buy dips to VWAP in expansion) and bear (sell rallies to VWAP in expansion).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate VWAP for 6h data
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Get daily data for Bollinger Band width calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate Bollinger Bands (20, 2)
    bb_middle = np.full(len(close_1d), np.nan)
    bb_std = np.full(len(close_1d), np.nan)
    
    for i in range(19, len(close_1d)):
        bb_middle[i] = np.mean(close_1d[i-19:i+1])
        bb_std[i] = np.std(close_1d[i-19:i+1])
    
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower  # Absolute width
    
    # BB width expansion/contraction: current width vs 10-period average
    bb_width_ma10 = np.full(len(bb_width), np.nan)
    for i in range(9, len(bb_width)):
        bb_width_ma10[i] = np.mean(bb_width[i-9:i+1])
    
    bb_width_ratio = bb_width / bb_width_ma10  # >1 = expanding, <1 = contracting
    bb_width_ratio_aligned = align_htf_to_ltf(prices, df_1d, bb_width_ratio)
    
    # VWAP deviation: how far price is from VWAP (as % of price)
    vwap_deviation = (close - vwap) / vwap  # Positive = above VWAP, negative = below VWAP
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_confirmed = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(30, 20)  # VWAP needs some data, BB needs 20, vol needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(vwap[i]) or 
            np.isnan(bb_width_ratio_aligned[i]) or
            np.isnan(vwap_deviation[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when BB width is expanding (>1.05)
        volatility_expanding = bb_width_ratio_aligned[i] > 1.05
        
        if position == 0:
            # Long entry: price below VWAP, volatility expanding, volume confirmed
            if (vwap_deviation[i] < -0.005 and  # 0.5% below VWAP
                volatility_expanding and
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price above VWAP, volatility expanding, volume confirmed
            elif (vwap_deviation[i] > 0.005 and   # 0.5% above VWAP
                  volatility_expanding and
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses above VWAP or volatility contracts
            if (vwap_deviation[i] > 0 or  # Price at or above VWAP
                bb_width_ratio_aligned[i] <= 1.02):  # Volatility contracting
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below VWAP or volatility contracts
            if (vwap_deviation[i] < 0 or   # Price at or below VWAP
                bb_width_ratio_aligned[i] <= 1.02):  # Volatility contracting
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VWAP_BBWidth_Expansion_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0