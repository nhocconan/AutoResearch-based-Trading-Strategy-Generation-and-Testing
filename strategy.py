#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (L3/H3) breakout with 1d volume spike and choppiness regime filter
# - Uses 1d Camarilla levels (L3, H3) as key support/resistance from prior day
# - Entry: 4h close breaks above H3 (long) or below L3 (short) with volume confirmation
# - Volume: 4h volume > 2.0x 20-period average to ensure breakout strength
# - Regime filter: 1d Choppiness Index > 61.8 (ranging market) for mean reversion at extremes
# - Exit: Opposite Camarilla level touch (L3 for long exit, H3 for short exit) or close back inside H3/L3
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) - avoids overtrading
# - Works in bull/bear: Camarilla levels adapt to volatility, chop filter avoids false breakouts in strong trends

name = "4h_1d_camarilla_vol_chop_v1"
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
    
    # Pre-compute 1d indicators for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR for Camarilla formula
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=5, min_periods=5).mean().values  # ATR(5) for Camarilla
    
    # Camarilla levels: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_1d + (1.1 * (high_1d - low_1d) / 2)
    camarilla_l3 = close_1d - (1.1 * (high_1d - low_1d) / 2)
    
    # 1d Choppiness Index for regime filter (chop > 61.8 = ranging)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) * (highest_high - lowest_low))
    atr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    # Avoid division by zero
    choppiness = np.zeros_like(atr14_1d)
    mask = (range_14 > 0) & (~np.isnan(range_14))
    choppiness[mask] = 100 * np.log10(sum_atr14[mask] / np.log10(14)) / np.log10(range_14[mask])
    chop_regime = choppiness > 61.8  # Chop > 61.8 = ranging market (good for mean reversion)
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    # 4h price data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h volume > 2.0x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(chop_regime_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches L3 or closes back inside H3
            if low[i] <= camarilla_l3_aligned[i] or close[i] < camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches H3 or closes back inside L3
            if high[i] >= camarilla_h3_aligned[i] or close[i] > camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and chop regime filter
            # Long: close breaks above H3 AND volume spike AND chop regime (ranging)
            if close[i] > camarilla_h3_aligned[i] and volume_spike[i] and chop_regime_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: close breaks below L3 AND volume spike AND chop regime (ranging)
            elif close[i] < camarilla_l3_aligned[i] and volume_spike[i] and chop_regime_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals