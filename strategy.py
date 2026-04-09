#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Elder Ray (Bull/Bear Power) with volume confirmation
# - Uses 6h Williams %R(14) for oversold/overbought signals (long < -80, short > -20)
# - Confirms with 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Requires volume > 1.5x 20-period average on 6t for institutional participation
# - Exits when Williams %R reverts to midpoint (-50) or opposite extreme
# - Position size: 0.25 (25% of capital) for controlled risk
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)
# - Williams %R captures short-term exhaustion, Elder Ray confirms trend strength via power

name = "6h_1d_williams_elderray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA13 for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13 = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d Elder Ray components
    bull_power = high_1d - ema13  # Bull Power: High - EMA13
    bear_power = low_1d - ema13   # Bear Power: Low - EMA13
    
    # 1d volume > 1.5x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * avg_volume_20)
    
    # Align 1d indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R reverts to midpoint (-50) or becomes overbought
            if williams_r[i] >= -50:  # Return to midpoint or overbought
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R reverts to midpoint (-50) or becomes oversold
            if williams_r[i] <= -50:  # Return to midpoint or oversold
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with Elder Ray confirmation and volume
            if (williams_r[i] <= -80 and          # Oversold
                bull_power_aligned[i] > 0 and     # Bullish power positive (uptrend strength)
                volume_spike_1d_aligned[i]):      # Volume confirmation
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] >= -20 and        # Overbought
                  bear_power_aligned[i] < 0 and   # Bearish power negative (downtrend strength)
                  volume_spike_1d_aligned[i]):    # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals