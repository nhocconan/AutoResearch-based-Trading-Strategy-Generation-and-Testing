#!/usr/bin/env python3
name = "6h_MarketFacilitation_Index_1dTrend_VolumeSpike"
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
    volume = prices['volume'].values
    
    # === 1d Data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d EMA34 for trend ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Market Facilitation Index (MFI) (6h) ===
    # MFI = (High - Low) / Volume
    range_hl = high - low
    mfi = np.where(volume > 0, range_hl / volume, 0)
    
    # === MFI smoothing (3-period SMA) ===
    mfi_sma = pd.Series(mfi).rolling(window=3, min_periods=3).mean().values
    
    # === Volume spike detection (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(mfi_sma[i]) or
            np.isnan(mfi_sma[i-1] if i > 0 else np.nan)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: MFI rising (increasing efficiency) + volume spike + 1d trend up
            if (mfi_sma[i] > mfi_sma[i-1] and 
                volume_spike[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: MFI falling (decreasing efficiency) + volume spike + 1d trend down
            elif (mfi_sma[i] < mfi_sma[i-1] and 
                  volume_spike[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: MFI falls or trend breaks
            if mfi_sma[i] < mfi_sma[i-1] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: MFI rises or trend breaks
            if mfi_sma[i] > mfi_sma[i-1] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals