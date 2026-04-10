#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
# - Camarilla pivots calculated from 1d OHLC: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
# - R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
# - Long when price breaks above R4 with volume confirmation (volume > 1.5x 20-period avg)
# - Short when price breaks below S4 with volume confirmation
# - Exit when price returns to R3/S3 levels (mean reversion) or opposite breakout occurs
# - Uses 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# - Camarilla levels from 1d provide institutional support/resistance that works in both bull and bear markets
# - Volume confirmation reduces false breakouts
# - Discrete position sizing (0.25) to minimize fee churn

name = "6h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # R4 = C + (H-L)*1.1, R3 = C + (H-L)*1.1/2
    # S4 = C - (H-L)*1.1, S3 = C - (H-L)*1.1/2
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to R3 (mean reversion) or breaks below S4 (reverse)
            if (prices['close'].iloc[i] <= camarilla_r3_aligned[i] or 
                prices['close'].iloc[i] < camarilla_s4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to S3 (mean reversion) or breaks above R4 (reverse)
            if (prices['close'].iloc[i] >= camarilla_s3_aligned[i] or 
                prices['close'].iloc[i] > camarilla_r4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout with volume confirmation
            if vol_spike_1d_aligned[i]:
                # Long signal: price breaks above R4
                if prices['close'].iloc[i] > camarilla_r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short signal: price breaks below S4
                elif prices['close'].iloc[i] < camarilla_s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals