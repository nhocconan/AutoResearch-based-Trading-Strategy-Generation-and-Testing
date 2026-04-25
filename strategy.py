#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_VolumeRegime_HTFTrend
Hypothesis: Daily Camarilla pivot (PP) breakout with volume confirmation and regime filter.
Long when price closes above PP with volume spike in bullish regime (weekly close > weekly open).
Short when price closes below PP with volume spike in bearish regime (weekly close < weekly open).
Exit when price re-enters Camarilla H3/L3 range.
Designed for low trade frequency (<20/year) on 1d timeframe, effective in both bull and bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # H3 = C + (H - L) * 1.1 / 4
    # L3 = C - (H - L) * 1.1 / 4
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    h3 = prev_close + range_hl * 1.1 / 4
    l3 = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (same timeframe, so direct use with 1-bar lag)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Get weekly data for regime filter (weekly bull/bear based on close vs open)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_bull = close_1w > open_1w  # True for bullish weekly candle
    weekly_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_bull.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(weekly_bull_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        weekly_regime_bull = weekly_bull_aligned[i] > 0.5  # True for bullish regime
        
        if position == 0:
            # Regime-based entry logic
            if weekly_regime_bull:  # Bullish weekly regime
                # Long: close above pivot with volume spike
                long_signal = (close[i] > pivot_aligned[i]) and vol_spike[i]
                # Short: close below pivot only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < s1_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Bearish weekly regime
                # Short: close below pivot with volume spike
                short_signal = (close[i] < pivot_aligned[i]) and vol_spike[i]
                # Long: close above pivot only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > r1_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: re-enter H3-L3 range
            exit_signal = (close[i] < h3_aligned[i] and close[i] > l3_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter H3-L3 range
            exit_signal = (close[i] > l3_aligned[i] and close[i] < h3_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_VolumeRegime_HTFTrend"
timeframe = "1d"
leverage = 1.0