#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and 1d chop regime filter
# - Long when price breaks above Camarilla H3 level + volume > 2.0x 20-period 1d volume SMA + CHOP(14) < 40 (trending)
# - Short when price breaks below Camarilla L3 level + volume > 2.0x 20-period 1d volume SMA + CHOP(14) < 40 (trending)
# - Exit: price returns to Camarilla Pivot point (mean reversion to equilibrium)
# - Position sizing: 0.25 discrete level
# - Camarilla levels identify intraday support/resistance from prior day
# - Volume conviction filters breakouts, chop regime avoids false signals in ranging markets
# - Works in bull/bear: breakouts in both directions, chop filter ensures trending environment

name = "4h_1d_camarilla_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # Camarilla: based on prior day's high, low, close
    h1 = df_1d['high'].values
    l1 = df_1d['low'].values
    c1 = df_1d['close'].values
    
    # Prior day's values (shifted by 1 to avoid look-ahead)
    high_prev = np.roll(h1, 1)
    low_prev = np.roll(l1, 1)
    close_prev = np.roll(c1, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Camarilla calculations
    range_prev = high_prev - low_prev
    camarilla_pivot = (high_prev + low_prev + close_prev) / 3.0
    camarilla_h3 = camarilla_pivot + (range_prev * 1.1 / 4.0)
    camarilla_l3 = camarilla_pivot - (range_prev * 1.1 / 4.0)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume SMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate Choppiness Index (CHOP) on 1d timeframe
    # CHOP = 100 * log10(sum(TR over n) / (n * (max_high - min_low))) / log10(n)
    tr_1d = np.maximum(h1 - l1, np.maximum(np.abs(h1 - np.roll(c1, 1)), np.abs(l1 - np.roll(c1, 1))))
    tr_1d[0] = h1[0] - l1[0]  # first bar TR
    
    # True Range sum over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over 14 periods
    max_high_14 = pd.Series(h1).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(l1).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula
    chop_denominator = 14 * (max_high_14 - min_low_14)
    chop_ratio = np.where(chop_denominator > 0, tr_sum_14 / chop_denominator, 0)
    chop_1d = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_1d = np.where((chop_denominator > 0) & (chop_ratio > 0), chop_1d, 50)  # default to 50 when invalid
    
    # Align CHOP to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: CHOP < 40 indicates trending market (not choppy/ranging)
        regime_filter = chop_1d_aligned[i] < 40
        
        # Camarilla breakout entry conditions
        # Long: price breaks above H3 + volume confirmation + trending regime
        # Short: price breaks below L3 + volume confirmation + trending regime
        long_entry = (close[i] > camarilla_h3_aligned[i] and 
                     vol_confirm and 
                     regime_filter)
        short_entry = (close[i] < camarilla_l3_aligned[i] and 
                      vol_confirm and 
                      regime_filter)
        
        # Exit conditions: price returns to Camarilla Pivot point (mean reversion)
        exit_long = close[i] < camarilla_pivot_aligned[i]
        exit_short = close[i] > camarilla_pivot_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals