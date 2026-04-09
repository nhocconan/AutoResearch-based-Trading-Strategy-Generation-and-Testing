#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot L3/H3 breakout + 12h volume spike + choppiness regime filter
# - Primary signal: Break of Camarilla L3 (long) or H3 (short) from prior 12h bar
# - Volume confirmation: 4h volume > 1.5x 20-period median volume (avoid low-participation signals)
# - Regime filter: Choppiness Index(14) between 38.2 and 61.8 (avoid strong trends where breakouts fail)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Camarilla levels provide intraday structure, volume confirms participation,
#   chop filter avoids whipsaws in strong trends, effective in ranging and trending markets

name = "4h_12h_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h indicators for Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for prior 12h bar
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_12h + 1.1 * (high_12h - low_12h) / 4
    camarilla_l3 = close_12h - 1.1 * (high_12h - low_12h) / 4
    
    # Align 12h Camarilla levels to 4h timeframe (completed 12h bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h volume regime: volume > 1.5x 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    # 4h Choppiness Index(14) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high)-min(low))))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(14) * (max_high_14 - min_low_14)
    chop = np.where(
        (chop_denominator > 0) & (atr_14 > 0),
        100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / chop_denominator),
        50.0  # neutral when undefined
    )
    # Chop regime: 38.2 < CHOP < 61.8 (avoid strong trends)
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
            # Exit: price re-enters Camarilla H3-L3 range OR chop regime breaks down
            if (close[i] <= camarilla_h3_aligned[i] and close[i] >= camarilla_l3_aligned[i]) or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Camarilla H3-L3 range OR chop regime breaks down
            if (close[i] <= camarilla_h3_aligned[i] and close[i] >= camarilla_l3_aligned[i]) or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and chop regime
            # Long: price breaks above Camarilla H3 AND volume regime AND chop regime
            if close[i] > camarilla_h3_aligned[i] and volume_regime[i] and chop_regime[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Camarilla L3 AND volume regime AND chop regime
            elif close[i] < camarilla_l3_aligned[i] and volume_regime[i] and chop_regime[i]:
                position = -1
                signals[i] = -0.25
    
    return signals