# 2025-06-08
# Hypothesis: 6h EMA crossover with 1d volume confirmation and 1d volatility filter
# Uses EMA(13) and EMA(34) crossover for trend detection on 6h timeframe
# Confirms with 1d volume spike (2x 20-period average) to ensure institutional participation
# Filters out low volatility periods using 1d ATR ratio (< 0.5 indicates chop)
# Designed for 5-15 trades per year (~20-60 total over 4 years) to minimize fee drag
# Works in both bull and bear markets by requiring volume confirmation and volatility filter

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_EMA13_34_1dVol_VolFilter"
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
    
    # Calculate EMA crossover on 6h
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for volume and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Volume spike detection on 1d (2x 20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (vol_ma_20 * 2.0)
    vol_spike_6h = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Volatility filter: ATR ratio < 0.5 indicates choppy market
    # Calculate ATR(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr[0] = high_1d[0] - low_1d[0]  # First period
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # ATR ratio: current ATR / 20-period average ATR
    atr_ma_20 = pd.Series(atr14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr14 / atr_ma_20
    # Filter: only trade when volatility is elevated (ratio > 0.5)
    vol_filter = atr_ratio > 0.5
    vol_filter_6h = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13[i]) or np.isnan(ema34[i]) or 
            np.isnan(vol_spike_6h[i]) or np.isnan(vol_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: EMA13 crosses above EMA34, volume spike, volatility filter
            if ema13[i] > ema34[i] and ema13[i-1] <= ema34[i-1] and vol_spike_6h[i] and vol_filter_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: EMA13 crosses below EMA34, volume spike, volatility filter
            elif ema13[i] < ema34[i] and ema13[i-1] >= ema34[i-1] and vol_spike_6h[i] and vol_filter_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: EMA13 crosses below EMA34 or volatility drops
            if ema13[i] < ema34[i] or not vol_filter_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: EMA13 crosses above EMA34 or volatility drops
            if ema13[i] > ema34[i] or not vol_filter_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals