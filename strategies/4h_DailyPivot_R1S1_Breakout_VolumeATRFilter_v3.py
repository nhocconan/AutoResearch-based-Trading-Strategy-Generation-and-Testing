#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily pivot-based breakout with volume confirmation and ATR volatility filter.
# Uses daily pivot levels (R1/S1) as dynamic support/resistance on 4h chart.
# Requires volume spike (>2x 24-period average) and minimum ATR volatility to avoid false breakouts in low volatility.
# Exit when price returns to daily pivot (mean reversion to equilibrium).
# Designed for fewer trades (<150/year) with clear entry/exit rules to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in direction of momentum with volatility filter.
name = "4h_DailyPivot_R1S1_Breakout_VolumeATRFilter_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for pivot and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Previous daily OHLC
    prev_close_d = df_1d['close'].shift(1).values
    prev_high_d = df_1d['high'].shift(1).values
    prev_low_d = df_1d['low'].shift(1).values
    
    # Pivot levels: R1, S1
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3
    range_d = prev_high_d - prev_low_d
    R1_d = pivot_d + range_d
    S1_d = pivot_d - range_d
    
    # ATR(14) for stop filter
    tr1 = prev_high_d - prev_low_d
    tr2 = np.abs(prev_high_d - prev_close_d)
    tr3 = np.abs(prev_low_d - prev_close_d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align to 4h
    R1_d_aligned = align_htf_to_ltf(prices, df_1d, R1_d)
    S1_d_aligned = align_htf_to_ltf(prices, df_1d, S1_d)
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    atr_d_aligned = align_htf_to_ltf(prices, df_1d, atr_d)
    
    # Volume filter: current volume > 2.0 * 24-period average (24 * 4h = 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_d_aligned[i]) or np.isnan(S1_d_aligned[i]) or
            np.isnan(pivot_d_aligned[i]) or np.isnan(atr_d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_val = R1_d_aligned[i]
        S1_val = S1_d_aligned[i]
        pivot_val = pivot_d_aligned[i]
        atr_val = atr_d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume and ATR filter (avoid low-vol breakouts)
            if close_val > R1_val and vol_filter and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and ATR filter
            elif close_val < S1_val and vol_filter and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below pivot
            if close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above pivot
            if close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals