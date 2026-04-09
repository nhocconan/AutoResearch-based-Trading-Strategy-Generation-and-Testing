#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume regime filter
# - Primary signal: 6h price breaks above Camarilla R4 or below S4 from prior 1d session
# - Continuation filter: 1d volume > 20-period median volume (ensures participation)
# - Exit: price retreats to Camarilla R3/S3 levels (profit target) or opposite S4/R4 (stop)
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Breakouts capture strong moves, volume filter avoids false signals in low-participation environments

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels (based on prior day's high, low, close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # Using prior day's values to avoid look-ahead
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    
    # First day will have NaN due to roll, that's correct (no prior day)
    rang = prior_high - prior_low
    camarilla_r4 = prior_close + rang * 1.1 / 2
    camarilla_r3 = prior_close + rang * 1.1 / 4
    camarilla_s3 = prior_close - rang * 1.1 / 4
    camarilla_s4 = prior_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (values update only when new 1d bar forms)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 1d volume regime filter
    volume_1d = df_1d['volume'].values
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_regime_1d = volume_1d > median_volume_20
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d)
    
    # 6h price data
    close_6h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price retreats to R3 (profit target) or breaks below S4 (stop)
            if close_6h[i] <= camarilla_r3_aligned[i] or close_6h[i] < camarilla_s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retreats to S3 (profit target) or breaks above R4 (stop)
            if close_6h[i] >= camarilla_s3_aligned[i] or close_6h[i] > camarilla_r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation
            # Long: price breaks above R4 with volume regime
            if (close_6h[i] > camarilla_r4_aligned[i] and 
                volume_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with volume regime
            elif (close_6h[i] < camarilla_s4_aligned[i] and 
                  volume_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals