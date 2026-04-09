#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume regime filter
# - Primary signal: 6h price breaks above Camarilla R4 (long) or below S4 (short) from prior 1d
# - Volume filter: 1d volume > 20-period median volume (ensures participation)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Camarilla levels adapt to volatility, breakouts capture momentum,
#   volume filter ensures institutional participation reducing false breakouts

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
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
    volume_1d = df_1d['volume'].values
    
    # Calculate prior 1d Camarilla levels (using prior day's high/low/close)
    # Shift by 1 to avoid look-ahead: use previous day's data for today's levels
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    # First value will be invalid (rolled from end), but min_periods handles this
    
    # Camarilla levels calculation
    # R4 = Close + ((High - Low) * 1.1/2)
    # S4 = Close - ((High - Low) * 1.1/2)
    camarilla_range = prior_high - prior_low
    r4 = prior_close + (camarilla_range * 1.1 / 2)
    s4 = prior_close - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_regime = volume_1d > median_volume_20
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    # 6h price data
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below prior day's close (mean reversion) OR volatility contraction
            if close[i] < prior_close[i] or camarilla_range[i] < np.nanmedian(camarilla_range[max(0, i-20):i+1]) * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above prior day's close OR volatility contraction
            if close[i] > prior_close[i] or camarilla_range[i] < np.nanmedian(camarilla_range[max(0, i-20):i+1]) * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume regime
            # Long: price breaks above R4 with volume participation
            if close[i] > r4_aligned[i] and volume_regime_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with volume participation
            elif close[i] < s4_aligned[i] and volume_regime_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals