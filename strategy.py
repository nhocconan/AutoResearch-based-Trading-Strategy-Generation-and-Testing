# 6h Donchian(20) breakout + daily pivot direction + volume confirmation
# Hypothesis: Daily pivot points (PP, R1, S1) provide strong institutional support/resistance.
# Breakouts of 6h Donchian channels in direction of daily pivot bias capture
# institutional flow while avoiding false breakouts. Works in bull/bear by
# following daily bias. Volume confirmation ensures commitment.
# Target: 12-37 trades/year (50-150 total over 4 years)
# Position size: 0.25 (discrete to minimize churn)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for pivots and Donchian (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    
    # Align daily pivots to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 6h Donchian channel (20-period)
    lookback = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    for i in range(lookback-1, n):
        dc_upper[i] = np.max(high[i-lookback+1:i+1])
        dc_lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian upper + above daily R1 + volume spike
            if close[i] > dc_upper[i] and close[i] > r1_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower + below daily S1 + volume spike
            elif close[i] < dc_lower[i] and close[i] < s1_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses opposite Donchian level
            if position == 1:
                # Exit long: Close below Donchian lower
                if close[i] < dc_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Close above Donchian upper
                if close[i] > dc_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_DailyPivot_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0