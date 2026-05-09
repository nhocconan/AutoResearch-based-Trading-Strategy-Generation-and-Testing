#!/usr/bin/env python3
# 2025-06-22 | 4h_EquiVolume_SMA50_Trend_Trading
# Hypothesis: Trend-following strategy using 50-period SMA on EquiVolume-weighted price, with volume confirmation and ATR-based stop.
# EquiVolume combines price and volume to highlight strong moves with high participation. Trend filter (price > SMA50) ensures alignment with higher-timeframe momentum.
# Volume spike (>1.5x 20-period average) confirms breakout strength. Designed for low trade frequency (20-50/year) to minimize fee drag.
# Works in bull markets (trend continuation) and bear markets (avoids false breakouts via volume/trend filters).

name = "4h_EquiVolume_SMA50_Trend_Trading"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # Calculate Typical Price and EquiVolume
    typical_price = (high + low + close) / 3.0
    equi_volume = typical_price * volume  # Price * Volume
    
    # Calculate 50-period SMA of EquiVolume / Volume = VWAP-like but using SMA
    # EquiVolume SMA / Volume SMA approximates volume-weighted average
    vol_sum = np.full_like(volume, np.nan)
    eqi_sum = np.full_like(equi_volume, np.nan)
    
    if len(volume) >= 50:
        vol_sum[49] = np.sum(volume[0:50])
        eqi_sum[49] = np.sum(equi_volume[0:50])
        for i in range(50, len(volume)):
            vol_sum[i] = vol_sum[i-1] + volume[i] - volume[i-50]
            eqi_sum[i] = eqi_sum[i-1] + equi_volume[i] - equi_volume[i-50]
    
    sma_eqi_vol = np.full_like(close, np.nan)
    valid_vol = vol_sum != 0
    sma_eqi_vol[valid_vol] = eqi_sum[valid_vol] / vol_sum[valid_vol]
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(20, 50)  # Ensure volume MA and SMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_eqi_vol[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price above SMA(EquiVolume/Volume) AND volume spike
            if close[i] > sma_eqi_vol[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price below SMA(EquiVolume/Volume) AND volume spike
            elif close[i] < sma_eqi_vol[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit long: price below SMA OR volume drops (loss of momentum)
            if close[i] < sma_eqi_vol[i] or volume_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above SMA OR volume drops
            if close[i] > sma_eqi_vol[i] or volume_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals