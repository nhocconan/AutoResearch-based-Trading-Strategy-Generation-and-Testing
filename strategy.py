#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with daily volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND daily volume > 2x 20-day volume SMA AND chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND daily volume > 2x 20-day volume SMA AND chop > 61.8
# - Exit: price reverts to Camarilla Pivot point (midline) or opposite breakout with volume confirmation
# - Uses 1d HTF for Camarilla levels and chop filter to avoid look-ahead
# - Position sizing: 0.25 discrete level
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots work well in ranging markets (chop > 61.8) which are common in 2025 bear/range regime
# - Volume confirmation ensures breakouts have conviction, reducing false signals

name = "12h_1d_camarilla_volspike_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators for HTF
    # Daily OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous day)
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    # Pivot = (high + low + close)/3
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.0 * range_1d
    camarilla_l3 = close_1d - 1.0 * range_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 12h timeframe (with completed bar delay)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Daily volume spike filter
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_sma_20_1d
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Daily choppiness regime filter (CHOP > 61.8 = ranging market)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (max(high)-min(low)))
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(max_high_14 - min_low_14)
    # Avoid division by zero or log of zero
    chop_denominator = np.where((max_high_14 - min_low_14) > 0, chop_denominator, 1)
    chop_1d = 100 * (np.log10(sum_atr_14) / chop_denominator)
    chop_filter_1d = chop_1d > 61.8  # Ranging market regime
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter_1d)
    
    # 12h volume confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_sma_20
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_filter_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # HTF conditions (must all be true)
        htf_conditions = (volume_spike_aligned[i] and 
                         chop_filter_aligned[i])
        
        # 12h volume confirmation
        vol_cond = volume_confirm[i]
        
        if position == 0:  # Flat - look for entry
            if htf_conditions and vol_cond:
                # Long breakout above H3
                if close[i] > camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown below L3
                elif close[i] < camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when price returns to pivot point or breaks below L3 with volume
            exit_condition = (close[i] < camarilla_pivot_aligned[i]) or \
                           (close[i] < camarilla_l3_aligned[i] and vol_cond)
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when price returns to pivot point or breaks above H3 with volume
            exit_condition = (close[i] > camarilla_pivot_aligned[i]) or \
                           (close[i] > camarilla_h3_aligned[i] and vol_cond)
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals