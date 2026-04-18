#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels (based on previous day's OHLC)
    # Using current day's OHLC for next day's levels (standard practice)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_hl * 1.1 / 12)
    R2 = pivot + (range_hl * 1.1 / 6)
    R3 = pivot + (range_hl * 1.1 / 4)
    R4 = pivot + (range_hl * 1.1 / 2)
    S1 = pivot - (range_hl * 1.1 / 12)
    S2 = pivot - (range_hl * 1.1 / 6)
    S3 = pivot - (range_hl * 1.1 / 4)
    S4 = pivot - (range_hl * 1.1 / 2)
    
    # Align daily levels to 6h timeframe (wait for daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # ATR filter: only trade when ATR(14) > 0.5 * ATR(50) (avoid choppy markets)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_filter = (atr_14 > 0) & (atr_50 > 0) & (atr_14 > 0.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_filter = volume_filter[i]
        atr_filt = atr_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume and ATR confirmation
            if close_val > R1_aligned[i] and vol_filter and atr_filt:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and ATR confirmation
            elif close_val < S1_aligned[i] and vol_filter and atr_filt:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1
            if close_val < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1
            if close_val > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals