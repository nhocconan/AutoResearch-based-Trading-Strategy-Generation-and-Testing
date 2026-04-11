#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (standard multipliers)
    r4_1d = close_1d + range_1d * 1.1 / 2
    r3_1d = close_1d + range_1d * 1.1 / 4
    r2_1d = close_1d + range_1d * 1.1 / 6
    r1_1d = close_1d + range_1d * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12
    s2_1d = close_1d - range_1d * 1.1 / 6
    s3_1d = close_1d - range_1d * 1.1 / 4
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Align daily pivots to 12h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_30[i]
        
        # Camarilla-based signals
        long_signal = False
        short_signal = False
        
        # Long: price breaks above R4 with volume
        if price_close > r4 and volume_confirmed:
            long_signal = True
        
        # Short: price breaks below S4 with volume
        if price_close < s4 and volume_confirmed:
            short_signal = True
        
        # Exit conditions: return to daily pivot
        pivot_1d_val = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3 if len(high_1d) > 0 else 0
        pivot_array = np.full_like(high_1d, pivot_1d_val)
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_array)[i]
        
        exit_long = price_close < pivot_aligned
        exit_short = price_close > pivot_aligned
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h Camarilla breakout strategy with volume confirmation and daily pivot exit.
# Enters long when price breaks above R4 (strong bullish breakout) with volume confirmation.
# Enters short when price breaks below S4 (strong bearish breakdown) with volume confirmation.
# Exits when price returns to daily pivot point (mean reversion to equilibrium).
# Uses volume confirmation (>1.5x 30-period average) to ensure institutional participation.
# Target: 15-25 trades per year to minimize fee decay while capturing strong directional moves.
# Works in both bull and bear markets by trading breakouts in either direction.