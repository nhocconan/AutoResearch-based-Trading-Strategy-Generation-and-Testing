#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d volume-weighted average price (VWAP) and volume imbalance
# - Long when price closes above 1d VWAP with increasing volume and bullish volume imbalance
# - Short when price closes below 1d VWAP with increasing volume and bearish volume imbalance
# - Exit when price crosses back below/above 1d VWAP
# - Uses volume-weighted price to capture institutional activity and volume imbalance to detect smart money
# - Designed to work in both bull and bear markets by following volume-confirmed trends
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_VWAP_VolumeImbalance_1d"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d typical price and cumulative values for VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tpv_1d = typical_price_1d * df_1d['volume']
    
    # Cumulative sums for VWAP calculation
    cum_tpv_1d = np.cumsum(tpv_1d.values)
    cum_volume_1d = np.cumsum(df_1d['volume'].values)
    
    # Avoid division by zero
    vwap_1d = np.where(cum_volume_1d != 0, cum_tpv_1d / cum_volume_1d, np.nan)
    
    # Align 1d VWAP to 6h timeframe
    vwap_1d_6h = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume imbalance: (buy volume - sell volume) / total volume
    # Approximate using price close relative to VWAP and volume
    vw_diff = close - vwap_1d_6h
    volume_imbalance = np.where(vw_diff > 0, volume, -volume)
    volume_imbalance_ma = pd.Series(volume_imbalance).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: increasing volume trend
    vol_ma_10 = pd.Series(volume).ewm(span=10, adjust=False, min_periods=10).mean().values
    vol_increasing = volume > vol_ma_10
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(vwap_1d_6h[i]) or np.isnan(volume_imbalance_ma[i]) or 
            np.isnan(vol_increasing[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above VWAP with bullish volume imbalance and increasing volume
            if close[i] > vwap_1d_6h[i] and volume_imbalance_ma[i] > 0 and vol_increasing[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP with bearish volume imbalance and increasing volume
            elif close[i] < vwap_1d_6h[i] and volume_imbalance_ma[i] < 0 and vol_increasing[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below VWAP
            if close[i] < vwap_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above VWAP
            if close[i] > vwap_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals