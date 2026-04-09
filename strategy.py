#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d EMA200 trend filter + volume confirmation
# - Primary signal: 12h close breaks above/below Camarilla H3/L3 levels from prior 1d session
# - Trend filter: 1d EMA200 - price must be above EMA for longs, below for shorts (avoid counter-trend)
# - Volume confirmation: 12h volume > 20-period median volume (ensure participation)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla pivots provide structure, EMA200 filter avoids major trend mistakes

name = "12h_1d_camarilla_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    # Pre-compute 1d indicators for Camarilla levels and EMA200
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, L3 (based on prior day)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # 1d EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
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
            np.isnan(ema_200_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR price crosses below EMA200
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR price crosses above EMA200
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and EMA200 filter
            # Long: close breaks above H3 AND volume regime AND price above EMA200
            if close[i] > camarilla_h3_aligned[i] and volume_regime[i] and close[i] > ema_200_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: close breaks below L3 AND volume regime AND price below EMA200
            elif close[i] < camarilla_l3_aligned[i] and volume_regime[i] and close[i] < ema_200_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals