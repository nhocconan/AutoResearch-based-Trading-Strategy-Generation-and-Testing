# 6h Pivot B1 S1 Breakout with Volume Filter
# Targets 12-37 trades/year by requiring breakouts of pivot-derived levels with volume confirmation.
# Works in bull/bear markets via price level respect and volume confirmation filter.
# Uses 12h for pivot calculation, 6h for entry timing, no look-ahead.

#!/usr/bin/env python3
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
    
    # === 12h data (HTF for pivot levels) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's pivot levels (B1, S1 from Camarilla)
    # Pivot = (H+L+C)/3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    # B1 = P + 1.1/12 * (H-L)  (bullish breakout level)
    # S1 = P - 1.1/12 * (H-L)  (bearish breakdown level)
    b1_12h = pivot_12h + 1.1/12.0 * range_12h
    s1_12h = pivot_12h - 1.1/12.0 * range_12h
    
    # Align to 6h timeframe (previous 12h bar's levels available after 12h bar closes)
    b1_12h_aligned = align_htf_to_ltf(prices, df_12h, b1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # === 6h indicators ===
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(b1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        b1 = b1_12h_aligned[i]
        s1 = s1_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S1 (breakdown)
            if price < s1:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above B1 (breakout)
            if price > b1:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above B1 with volume confirmation
                if (price > b1) and (vol_ratio_val > 2.0):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below S1 with volume confirmation
                elif (price < s1) and (vol_ratio_val > 2.0):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Pivot_B1_S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0