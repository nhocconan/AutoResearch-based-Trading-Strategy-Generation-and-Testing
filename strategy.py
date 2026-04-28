# 4H_CAMARILLA_R1_S1_BREAKOUT_1D_VOLUME_SPIKE
# Hypothesis: Use 1D Camarilla R1/S1 levels (tighter than R2/S2) with volume confirmation and session filter (08-20 UTC).
# Tighter levels should increase trade frequency to optimal range (20-50/year) while volume spike filters false breakouts.
# Works in bull/bear: Breakouts capture momentum, volume confirms institutional participation, session filter avoids low-liquidity periods.
# Target: 25-40 trades/year per symbol.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot points and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    hl_range = high_1d - low_1d
    r1_1d = close_1d + hl_range * 1.1 / 12.0
    s1_1d = close_1d - hl_range * 1.1 / 12.0
    
    # Align daily Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate daily volume spike (current volume > 1.8x 20-period MA)
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.8 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Daily Camarilla levels
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        
        # Volume spike confirmation from daily timeframe
        vol_spike = vol_spike_aligned[i] > 0.5
        
        # Entry conditions: 
        # Long: Price breaks above daily R1 with volume spike
        # Short: Price breaks below daily S1 with volume spike
        long_entry = (close[i] > r1) and vol_spike
        short_entry = (close[i] < s1) and vol_spike
        
        # Exit conditions: 
        # Long exit: price returns below daily midpoint (Camarilla close)
        # Short exit: price returns above daily midpoint
        midpoint_1d = (r1_1d + s1_1d) / 2.0  # This equals close_1d in Camarilla
        midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
        midpoint_val = midpoint_aligned[i]
        
        long_exit = close[i] < midpoint_val
        short_exit = close[i] > midpoint_val
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_1D_Volume_Spike"
timeframe = "4h"
leverage = 1.0