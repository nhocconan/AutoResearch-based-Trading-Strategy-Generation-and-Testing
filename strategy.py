# 12h_1d_RangeBreakout_VolumeConfirmation_v1
# Hypothesis: 12h price breaks daily range with volume confirmation.
# Long when price breaks above prior 12h high with volume > 1.5x average.
# Short when price breaks below prior 12h low with volume > 1.5x average.
# Uses 1d timeframe for volatility filter (ATR-based) to avoid choppy markets.
# Target: 15-30 trades/year per symbol.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for range calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar high/low (to avoid look-ahead)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    
    # Load 1d data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily timeframe
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to main timeframe
    prev_high_12h_aligned = align_htf_to_ltf(prices, df_12h, prev_high_12h)
    prev_low_12h_aligned = align_htf_to_ltf(prices, df_12h, prev_low_12h)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Main timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(prev_high_12h_aligned[i]) or np.isnan(prev_low_12h_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        high_i = high[i]
        low_i = low[i]
        prev_high_val = prev_high_12h_aligned[i]
        prev_low_val = prev_low_12h_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_ok = vol_filter[i]
        
        # Volatility filter: only trade when ATR > 0 (avoid dead markets)
        vol_filter_ok = atr_val > 0
        
        if position == 0:
            # Long: price breaks above prior 12h high with volume
            if high_i > prev_high_val and vol_ok and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior 12h low with volume
            elif low_i < prev_low_val and vol_ok and vol_filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below prior 12h low
            if low_i < prev_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above prior 12h high
            if high_i > prev_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_RangeBreakout_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0