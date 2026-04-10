#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d with volume confirmation
# - Camarilla levels (R3, R4, S3, S4) calculated from prior 1d OHLC
# - Long when price breaks above R4 with volume > 1.5x 20-period average
# - Short when price breaks below S4 with volume > 1.5x 20-period average
# - Exit when price retests the pivot point (PP) from 1d
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Camarilla levels work in both trending and ranging markets due to mathematical construction

name = "6h_1d_camarilla_breakout_v1"
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
    
    # Pre-compute 1d Camarilla levels (based on prior day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior day's Camarilla levels (shifted by 1 to avoid look-ahead)
    # Camarilla formulas:
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # S3 = PP - (H - L) * 1.1/4
    # S4 = PP - (H - L) * 1.1/2
    
    # Use prior day's data (shifted by 1) for current day's levels
    pp = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    rng = np.roll(high_1d, 1) - np.roll(low_1d, 1)
    r4 = pp + rng * 1.1 / 2
    r3 = pp + rng * 1.1 / 4
    s3 = pp - rng * 1.1 / 4
    s4 = pp - rng * 1.1 / 2
    
    # Handle first bar (no prior data)
    pp[0] = np.nan
    r4[0] = np.nan
    r3[0] = np.nan
    s3[0] = np.nan
    s4[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price retests pivot point (mean reversion)
            if prices['close'].iloc[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests pivot point (mean reversion)
            if prices['close'].iloc[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation
            if vol_spike[i]:
                close_price = prices['close'].iloc[i]
                # Breakout long: price closes above R4
                if close_price > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price closes below S4
                elif close_price < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals