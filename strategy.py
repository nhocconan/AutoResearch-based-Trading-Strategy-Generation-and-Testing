#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout + 4h volume filter + session filter
# - Primary signal: 1h price breaks above/below Camarilla H3/L3 levels from prior 4h bar
# - Trend filter: 4h volume > 20-period median volume (avoid low-participation breakouts)
# - Session filter: trade only 08:00-20:00 UTC to avoid Asian session noise
# - Position size: 0.20 (discrete level) to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines
# - Works in bull/bear: Camarilla levels adapt to volatility, volume filter ensures conviction

name = "1h_4h_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h Camarilla levels (H3, L3) from prior completed 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    camarilla_h3 = close_4h + (1.1 * (high_4h - low_4h) / 6)
    camarilla_l3 = close_4h - (1.1 * (high_4h - low_4h) / 6)
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar only)
    h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # 4h volume regime: volume > 20-period median volume
    volume_4h = df_4h['volume'].values
    median_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).median().values
    volume_regime_4h = volume_4h > median_volume_20
    volume_regime_aligned = align_htf_to_ltf(prices, df_4h, volume_regime_4h)
    
    # Session filter: 08:00-20:00 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or
            np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below prior 4h bar's L3 level
            if close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above prior 4h bar's H3 level
            if close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation
            # Long: price breaks above H3 with volume confirmation
            if (close[i] > h3_aligned[i] and 
                volume_regime_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below L3 with volume confirmation
            elif (close[i] < l3_aligned[i] and 
                  volume_regime_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals