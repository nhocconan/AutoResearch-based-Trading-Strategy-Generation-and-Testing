#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# - Uses 1d Camarilla pivot levels (H3/L3) for mean reversion entries
# - Confirms with 1d volume > 2.0x its 20-period average (strong participation)
# - Filters by 1d choppiness index: CHOP > 61.8 = ranging market (good for mean reversion)
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in bull markets (mean reversion in ranges) and bear markets (mean reversion in ranges)
# - Camarilla levels provide objective support/resistance in ranging conditions
# - Volume filter reduces false signals, chop filter ensures we're in ranging market

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ATR (needed for Camarilla calculation)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # 1d ATR(5) for Camarilla calculation (standard period)
    atr_1d = pd.Series(tr_1d).rolling(window=5, min_periods=5).mean().values
    
    # 1d Camarilla levels: H3/L3 (most significant levels for mean reversion)
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_1d + (1.1 * (high_1d - low_1d) / 2)
    camarilla_l3 = close_1d - (1.1 * (high_1d - low_1d) / 2)
    
    # 1d Volume > 2.0x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20)
    
    # 1d Choppiness Index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop_1d = np.where((highest_high_14 - lowest_low_14) > 0, chop_1d, 50.0)
    chop_1d = np.where(np.isnan(chop_1d), 50.0, chop_1d)
    
    # Chop > 61.8 = ranging market (good for mean reversion)
    chop_ranging = chop_1d > 61.8
    
    # Align 1d indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging.astype(float))
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_ranging_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when price reaches Camarilla H3 (take profit) or L3 (stop loss)
            if high[i] >= camarilla_h3_aligned[i] or low[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price reaches Camarilla L3 (take profit) or H3 (stop loss)
            if low[i] <= camarilla_l3_aligned[i] or high[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion at Camarilla levels with volume and chop confirmation
            if (low[i] <= camarilla_l3_aligned[i] and    # Price touches or breaks L3 (long setup)
                volume_spike_1d_aligned[i] and           # Volume confirmation
                chop_ranging_aligned[i]):                # Ranging market condition
                position = 1
                signals[i] = 0.25
            elif (high[i] >= camarilla_h3_aligned[i] and # Price touches or breaks H3 (short setup)
                  volume_spike_1d_aligned[i] and         # Volume confirmation
                  chop_ranging_aligned[i]):              # Ranging market condition
                position = -1
                signals[i] = -0.25
    
    return signals