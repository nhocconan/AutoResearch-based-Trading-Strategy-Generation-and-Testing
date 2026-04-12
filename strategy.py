#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Pullback_v1
Hypothesis: Trade pullbacks to Camarilla pivot levels on 12h timeframe with 1d volume confirmation.
Camarilla levels provide institutional support/resistance. Pullbacks reduce false breakouts.
Volume confirms institutional participation. Works in bull/bear by trading mean reversion at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Pullback_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # R2 = PP + (H - L) * 1.1/6
    # R1 = PP + (H - L) * 1.1/12
    # S1 = PP - (H - L) * 1.1/12
    # S2 = PP - (H - L) * 1.1/6
    # S3 = PP - (H - L) * 1.1/4
    # S4 = PP - (H - L) * 1.1/2
    
    # Use previous day's data to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pp = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    r4 = pp + range_hl * 1.1 / 2.0
    r3 = pp + range_hl * 1.1 / 4.0
    r2 = pp + range_hl * 1.1 / 6.0
    r1 = pp + range_hl * 1.1 / 12.0
    s1 = pp - range_hl * 1.1 / 12.0
    s2 = pp - range_hl * 1.1 / 6.0
    s3 = pp - range_hl * 1.1 / 4.0
    s4 = pp - range_hl * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === DAILY VOLUME CONFIRMATION ===
    # Volume > 1.3x 20-day average indicates institutional interest
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1d / np.where(volume_ma == 0, 1, volume_ma)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_ratio_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        volume_confirm = volume_ratio_aligned[i] > 1.3
        
        # LONG SETUP: Pullback to S1 or S2 with volume confirmation
        long_setup = volume_confirm and (
            (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i]) or  # Bounce off S1
            (low[i] <= s2_aligned[i] and close[i] > s2_aligned[i])    # Bounce off S2
        )
        
        # SHORT SETUP: Pullback to R1 or R2 with volume confirmation
        short_setup = volume_confirm and (
            (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]) or  # Rejection at R1
            (high[i] >= r2_aligned[i] and close[i] < r2_aligned[i])    # Rejection at R2
        )
        
        # EXIT: Return to midpoint (PP) or opposite level touch
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        exit_long = position == 1 and (
            close[i] >= pp_aligned[i] or  # Return to midpoint
            high[i] >= r1_aligned[i]      # Hit resistance
        )
        exit_short = position == -1 and (
            close[i] <= pp_aligned[i] or  # Return to midpoint
            low[i] <= s1_aligned[i]       # Hit support
        )
        
        # Execute trades
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals