#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week Pivot Point (CPR) and 1-day Supertrend for trend alignment, with 6h volume confirmation.
# Pivot Points provide institutional support/resistance levels; Supertrend filters trend direction.
# Enters long when price is above weekly CPR pivot and above daily Supertrend, with volume spike.
# Enters short when price is below weekly CPR pivot and below daily Supertrend, with volume spike.
# Uses tight conditions to limit trades (~20-30/year) and avoid overtrading. Works in bull/bear via trend filter.
name = "6h_1w_CPR_1d_Supertrend_Volume"
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
    
    # Get 1w data for CPR (Central Pivot Range) - calculated once
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # CPR: Pivot = (H+L+C)/3, BC = (H+L)/2, TC = (Pivot - BC) + Pivot
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    bc_1w = (high_1w + low_1w) / 2.0
    tc_1w = (pivot_1w - bc_1w) + pivot_1w
    top_cpr_1w = np.maximum(pivot_1w, tc_1w)
    bottom_cpr_1w = np.minimum(pivot_1w, bc_1w)
    
    # Align CPR levels to 6s timeframe
    top_cpr_1w_aligned = align_htf_to_ltf(prices, df_1w, top_cpr_1w)
    bottom_cpr_1w_aligned = align_htf_to_ltf(prices, df_1w, bottom_cpr_1w)
    
    # Get 1d data for Supertrend - calculated once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Supertrend calculation (ATR-based)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR
    atr = np.zeros_like(close_1d)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / attr_period
    
    # Supertrend upper/lower bands
    hl2 = (high_1d + low_1d) / 2.0
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Supertrend direction
    supertrend = np.zeros_like(close_1d)
    trend_up = np.ones_like(close_1d, dtype=bool)
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > upper_band[i-1]:
            trend_up[i] = True
        elif close_1d[i] < lower_band[i-1]:
            trend_up[i] = False
        else:
            trend_up[i] = trend_up[i-1]
            if trend_up[i] and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if not trend_up[i] and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if trend_up[i] else upper_band[i]
    
    # Align Supertrend to 6s timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(top_cpr_1w_aligned[i]) or np.isnan(bottom_cpr_1w_aligned[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly CPR top AND above daily Supertrend with volume
            if (close[i] > top_cpr_1w_aligned[i] and 
                close[i] > supertrend_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly CPR bottom AND below daily Supertrend with volume
            elif (close[i] < bottom_cpr_1w_aligned[i] and 
                  close[i] < supertrend_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly CPR bottom or Supertrend
            if close[i] < bottom_cpr_1w_aligned[i] or close[i] < supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly CPR top or Supertrend
            if close[i] > top_cpr_1w_aligned[i] or close[i] > supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals