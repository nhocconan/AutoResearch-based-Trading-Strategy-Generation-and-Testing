#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly pivot resistance/support breakout with volume confirmation and weekly ATR filter.
# Uses weekly pivot levels (R1, S1) from prior week as breakout levels.
# Weekly ATR filter ensures sufficient volatility to avoid choppy markets.
# Volume confirmation adds conviction to breakouts.
# Designed for very low trade frequency (<10/year) to minimize fee drag in 1d timeframe.
# Works in bull markets (breakouts above R1) and bear markets (breakouts below S1).
name = "1d_WeeklyPivot_R1S1_Breakout_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and ATR calculation (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    
    # Calculate weekly ATR (14-period) using Wilder's smoothing
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    atr_w = np.full_like(tr_w, np.nan)
    if len(tr_w) >= atr_period:
        atr_w[atr_period-1] = np.nanmean(tr_w[:atr_period])
        for i in range(atr_period, len(tr_w)):
            if not np.isnan(atr_w[i-1]) and not np.isnan(tr_w[i]):
                atr_w[i] = atr_w[i-1] * (1 - 1/atr_period) + tr_w[i] * (1/atr_period)
            else:
                atr_w[i] = np.nan
    
    # ATR multiplier for volatility filter
    atr_mult = 1.5
    atr_threshold_w = atr_w * atr_mult
    
    # Align weekly pivot levels and ATR threshold to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1w, atr_threshold_w)
    
    # Calculate 20-period average volume for confirmation (using daily data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(atr_threshold_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Volatility filter: current ATR threshold must be positive (sufficient volatility)
        vol_filter = not np.isnan(atr_threshold_aligned[i]) and atr_threshold_aligned[i] > 0
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirmation AND volatility filter
            long_breakout = close[i] > r1_aligned[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] < s1_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] > r1_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals