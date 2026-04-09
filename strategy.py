#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout + 4h volume regime + session filter (08-20 UTC)
# - Primary signal: 1h price breaks above/below Camarilla H3/L3 levels from prior 4h bar
# - Trend filter: 4h volume > 20-period median volume (avoid low-participation breakouts)
# - Session filter: Only trade 08-20 UTC to avoid Asian session noise
# - Position size: 0.20 (discrete level) to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) for 1h strategy
# - Works in bull/bear: Camarilla levels adapt to volatility, volume filter ensures quality breakouts

name = "1h_4h_camarilla_volume_session_v1"
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
    
    # Pre-compute 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).median().values
    volume_regime_4h = volume_4h > median_volume_20
    
    # Align 4h volume regime to 1h timeframe (completed 4h bar only)
    volume_regime_aligned = align_htf_to_ltf(prices, df_4h, volume_regime_4h)
    
    # Pre-compute Camarilla levels for each 4h bar
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h) / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_regime_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla H3 OR volume regime fails
            if close[i] < camarilla_h3_aligned[i] or not volume_regime_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla L3 OR volume regime fails
            if close[i] > camarilla_l3_aligned[i] or not volume_regime_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume regime confirmation
            # Long: price breaks above Camarilla H3 AND volume regime
            if close[i] > camarilla_h3_aligned[i] and volume_regime_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short: price breaks below Camarilla L3 AND volume regime
            elif close[i] < camarilla_l3_aligned[i] and volume_regime_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals