#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume confirmation + choppiness regime filter
# - Primary signal: 12h price breaks above/below 1d Camarilla pivot levels (H3/L3)
# - Volume confirmation: 12h volume > 20-period median volume (avoid low-participation breakouts)
# - Regime filter: 1d choppiness index > 61.8 (range market) for mean reversion at pivot levels
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla pivots act as support/resistance in ranging markets,
#   volume confirms breakout validity, chop filter ensures we trade in appropriate regimes

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
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Camarilla pivot levels (based on previous day)
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align 1d Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d Choppiness Index (CHOP) - range/trend regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.max(high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]))], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = np.where(
        (hh14 - ll14) == 0,
        50.0,  # neutral when range is zero
        100 * np.log10(np.sum(atr14) / np.log10(hh14 - ll14) / np.log10(14))
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below L3 (support) OR chop < 38.2 (trending regime)
            if close[i] < camarilla_l3_aligned[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above H3 (resistance) OR chop < 38.2 (trending regime)
            if close[i] > camarilla_h3_aligned[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and chop > 61.8 (range regime)
            # Long: price breaks above H3 (resistance becomes support) AND volume regime AND chop > 61.8
            if close[i] > camarilla_h3_aligned[i] and volume_regime[i] and chop_aligned[i] > 61.8:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below L3 (support becomes resistance) AND volume regime AND chop > 61.8
            elif close[i] < camarilla_l3_aligned[i] and volume_regime[i] and chop_aligned[i] > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals