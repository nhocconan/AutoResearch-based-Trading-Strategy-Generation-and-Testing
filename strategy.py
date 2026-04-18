#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index + Daily Pivot R1/S1 Breakout with Volume Confirmation
# Works in bull/bear: Choppiness identifies regime (trend/range), pivots provide structure,
# volume confirms breakout validity. Targets 15-30 trades/year to avoid fee drag.
name = "12h_ChoppinessIndex_Pivot_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Choppiness Index (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    sum_tr = np.sum(tr) if n > 0 else 1
    chop = 100 * np.log10(atr * atr_period / (max_high - min_low + 1e-10)) / np.log10(atr_period)
    chop = np.where((max_high - min_low) != 0, chop, 50)  # Default to neutral when no range
    
    # Load daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot and R1/S1 from previous daily bar
    prev_close_d = df_1d['close'].shift(1).values
    prev_high_d = df_1d['high'].shift(1).values
    prev_low_d = df_1d['low'].shift(1).values
    
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3
    range_d = prev_high_d - prev_low_d
    R1_d = pivot_d + (range_d * 1.1 / 2)  # R1 = pivot + 0.55*range
    S1_d = pivot_d - (range_d * 1.1 / 2)  # S1 = pivot - 0.55*range
    
    # Align daily R1/S1 to 12h (wait for daily close)
    R1_d_aligned = align_htf_to_ltf(prices, df_1d, R1_d)
    S1_d_aligned = align_htf_to_ltf(prices, df_1d, S1_d)
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    
    # Volume filter: current volume > 1.5 * 24-period average (12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop[i]) or np.isnan(R1_d_aligned[i]) or np.isnan(S1_d_aligned[i]) or
            np.isnan(pivot_d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        chop_val = chop[i]
        R1_val = R1_d_aligned[i]
        S1_val = S1_d_aligned[i]
        pivot_val = pivot_d_aligned[i]
        vol_filter = volume_filter[i]
        
        # Choppiness regime: < 38.2 = trending, > 61.8 = ranging
        # In trending markets, we follow breakouts; in ranging, we fade extremes
        if chop_val < 38.2:  # Trending regime
            if position == 0:
                # Long: break above R1 with volume
                if close_val > R1_val and vol_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: break below S1 with volume
                elif close_val < S1_val and vol_filter:
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
        else:  # Ranging market (chop >= 38.2) - fade at extremes
            if position == 0:
                # Long: pullback to S1 with volume
                if close_val < S1_val and vol_filter:
                    signals[i] = 0.20
                    position = 1
                # Short: pullback to R1 with volume
                elif close_val > R1_val and vol_filter:
                    signals[i] = -0.20
                    position = -1
            elif position == 1:
                # Long exit: price reaches pivot or shows weakness
                if close_val >= pivot_val or close_val < close[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Short exit: price reaches pivot or shows strength
                if close_val <= pivot_val or close_val > close[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals