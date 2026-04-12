#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_camarilla_pivot_breakout_volume
# Uses daily Camarilla pivot levels (support/resistance) as breakout levels on 12h chart.
# Long when price breaks above R3 with volume confirmation (volume > 1.5x 20-period avg).
# Short when price breaks below S3 with volume confirmation.
# Exits when price crosses the pivot point (mean reversion).
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
# Works in trending markets via breakouts and ranging markets via mean reversion to pivot.
# Focus on BTC/ETH as primary targets.

name = "12h_1d_camarilla_pivot_breakout_volume"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily typical price and range
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    high_low = df_1d['high'] - df_1d['low']
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = typical_price.values
    
    # Camarilla levels
    # R4 = PP + (H-L) * 1.1/2
    # R3 = PP + (H-L) * 1.1/4
    # R2 = PP + (H-L) * 1.1/6
    # R1 = PP + (H-L) * 1.1/12
    # S1 = PP - (H-L) * 1.1/12
    # S2 = PP - (H-L) * 1.1/6
    # S3 = PP - (H-L) * 1.1/4
    # S4 = PP - (H-L) * 1.1/2
    
    r3 = pp + high_low.values * 1.1 / 4.0
    s3 = pp - high_low.values * 1.1 / 4.0
    
    # Align daily Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume confirmation: volume > 1.5 * 20-period average (12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above R3
        if close[i] > r3_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below S3
        elif close[i] < s3_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses pivot point (mean reversion)
        elif position == 1 and close[i] <= pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals