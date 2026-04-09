#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Primary signal: Price breaks above/below Camarilla pivot levels (H3/L3) from 1d timeframe
# - Trend filter: 1w EMA50 - price must be above/below EMA for alignment with higher timeframe trend
# - Volume confirmation: 12h volume > 30-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla levels act as support/resistance in ranges, EMA50 filter ensures trend alignment

name = "12h_1d_1w_camarilla_pivot_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels (H3, L3, H4, L4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate ranges
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    h3 = pp + (range_hl * 1.1 / 4)
    l3 = pp - (range_hl * 1.1 / 4)
    h4 = pp + (range_hl * 1.1 / 2)
    l4 = pp - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h volume regime: volume > 30-period median volume
    volume = prices['volume'].values
    median_volume_30 = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    volume_regime = volume > median_volume_30
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below L3 OR closes below 1w EMA50
            if prices['close'].iloc[i] < l3_aligned[i] or prices['close'].iloc[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above H3 OR closes above 1w EMA50
            if prices['close'].iloc[i] > h3_aligned[i] or prices['close'].iloc[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and 1w EMA50 filter
            # Long: Price breaks above H3 AND volume regime AND price above 1w EMA50
            if (prices['close'].iloc[i] > h3_aligned[i] and 
                volume_regime[i] and 
                prices['close'].iloc[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Price breaks below L3 AND volume regime AND price below 1w EMA50
            elif (prices['close'].iloc[i] < l3_aligned[i] and 
                  volume_regime[i] and 
                  prices['close'].iloc[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals