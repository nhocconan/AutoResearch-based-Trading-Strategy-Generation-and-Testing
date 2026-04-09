#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels (L3/H3) from 1d + volume confirmation + chop regime filter
# - Primary signal: Price touches Camarilla L3 (long) or H3 (short) from prior 1d
# - Volume confirmation: 12h volume > 24-period median volume (avoid low-participation signals)
# - Regime filter: 12h Choppiness Index(14) between 38.2 and 61.8 (avoid strong trends, favor mean reversion)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla pivots act as mean reversion levels in ranging markets,
#   volume confirms participation, chop filter avoids whipsaws in strong trends

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior 1d Camarilla levels (H3, L3)
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume regime: volume > 24-period median volume
    median_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).median().values
    volume_regime = volume > median_volume_24
    
    # 12h Choppiness Index(14)
    # Chop = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no prior close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop_denominator = np.log10(14) * (highest_high_14 - lowest_low_14)
    chop_denominator = np.where(chop_denominator == 0, 1, chop_denominator)
    chop = 100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / chop_denominator)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral when undefined
    
    # Chop regime: 38.2 < chop < 61.8 (ranging market, good for mean reversion)
    chop_regime = (chop > 38.2) & (chop < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_regime[i]) or
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses above Camarilla H3 (take profit) OR chop exits regime
            if close[i] >= camarilla_h3_aligned[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below Camarilla L3 (take profit) OR chop exits regime
            if close[i] <= camarilla_l3_aligned[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for price touching Camarilla levels with volume confirmation and chop regime
            # Long: price touches or crosses above Camarilla L3 from below
            if (low[i] <= camarilla_l3_aligned[i] and close[i] > camarilla_l3_aligned[i]) and \
               volume_regime[i] and chop_regime[i]:
                position = 1
                signals[i] = 0.25
            # Short: price touches or crosses below Camarilla H3 from above
            elif (high[i] >= camarilla_h3_aligned[i] and close[i] < camarilla_h3_aligned[i]) and \
                 volume_regime[i] and chop_regime[i]:
                position = -1
                signals[i] = -0.25
    
    return signals