#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + volume spike + choppiness regime filter
# - Primary signal: 4h close breaks above Camarilla H3 (long) or below L3 (short) from prior 1d
# - Volume confirmation: 4h volume > 1.5 * 20-period median volume (ensures participation)
# - Regime filter: Choppiness Index(14) > 61.8 (range market) for mean reversion at pivots
# - Position size: 0.25 (discrete level) to balance return and fee drag
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Camarilla levels adapt to volatility, chop filter avoids false breakouts in trends

name = "4h_1d_camarilla_breakout_chop_volume_v1"
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
    
    # Calculate Camarilla levels from prior 1d (H3, L3, H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #            L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_h4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_l4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Align 1d Camarilla levels to 4h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h volume regime: volume > 1.5 * 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * median_volume_20)
    
    # 4h Choppiness Index(14): measures trend vs range (0-100, >61.8 = range)
    # CHOP = 100 * LOG10(SUM(ATR(1)) / (MAX(HIGH)-MIN(LOW))) / LOG10(14)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.append([np.nan], close[:-1])), np.abs(low - np.append([np.nan], close[:-1]))))
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_atr1_14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_14 - min_low_14
    chop = np.where(
        (chop_denom == 0) | (sum_atr1_14 <= 0),
        50.0,  # neutral when no range or no ATR
        100 * np.log10(sum_atr1_14 / chop_denom) / np.log10(14)
    )
    chop_range = chop > 61.8  # range regime
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_range[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: close below H3 (failed breakout) OR chop regime ends (trend start)
            if close[i] < h3_aligned[i] or not chop_range[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above L3 (failed breakdown) OR chop regime ends (trend start)
            if close[i] > l3_aligned[i] or not chop_range[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and chop filter
            # Long: close > H3 AND volume spike AND chop range (mean reversion setup)
            if close[i] > h3_aligned[i] and volume_spike[i] and chop_range[i]:
                position = 1
                signals[i] = 0.25
            # Short: close < L3 AND volume spike AND chop range (mean reversion setup)
            elif close[i] < l3_aligned[i] and volume_spike[i] and chop_range[i]:
                position = -1
                signals[i] = -0.25
    
    return signals