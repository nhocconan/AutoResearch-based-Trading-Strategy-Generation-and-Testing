#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_pivot_volume_v2
# Uses daily Camarilla pivot levels (L3, H3) as entry triggers with volume confirmation.
# Long when price crosses above H3 with volume > 1.5x 20-period average.
# Short when price crosses below L3 with volume > 1.5x 20-period average.
# Exits when price returns to the daily pivot point (mean reversion).
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and ranging markets via mean reversion to pivot.
# Focus on BTC/ETH as primary targets.

name = "4h_1d_camarilla_pivot_volume_v2"
timeframe = "4h"
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
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_ = high_1d - low_1d
    # Camarilla levels
    # H3 = C + (H-L) * 1.1/2
    # L3 = C - (H-L) * 1.1/2
    camarilla_h3 = close_1d + range_ * 1.1 / 2.0
    camarilla_l3 = close_1d - range_ * 1.1 / 2.0
    # Pivot for exit
    camarilla_pivot = pivot
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: volume > 1.5 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]):
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
        
        # Long signal: price crosses above H3
        if close[i] > camarilla_h3_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price crosses below L3
        elif close[i] < camarilla_l3_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to pivot point (mean reversion)
        elif position == 1 and close[i] <= camarilla_pivot_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= camarilla_pivot_aligned[i]:
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