#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot levels from weekly data with volume confirmation
# Weekly Camarilla levels (R3/S3, R4/S4) provide institutional support/resistance with proven edge
# Fade at R3/S3 (mean reversion) and breakout continuation at R4/S4 (trend following)
# Volume confirmation (current 6h volume > 2.0x 20-period average) filters false signals
# Designed for 6h timeframe targeting 15-25 trades/year (60-100 over 4 years)
# Works in bull/bear: price reacts to weekly structure, volume confirms validity

name = "6h_1w_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    # Weekly range
    weekly_range = high_1w - low_1w
    # Camarilla levels based on previous weekly close
    camarilla_h4 = close_1w + (weekly_range * 1.1 / 2)  # R4
    camarilla_l4 = close_1w - (weekly_range * 1.1 / 2)  # S4
    camarilla_h3 = close_1w + (weekly_range * 1.1 / 4)  # R3
    camarilla_l3 = close_1w - (weekly_range * 1.1 / 4)  # S3
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average 6h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit conditions: stop at S3 or target at R4
            if close[i] <= camarilla_l3_aligned[i] or close[i] >= camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: stop at R3 or target at S4
            if close[i] >= camarilla_h3_aligned[i] or close[i] <= camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirmed:
                # Fade at R3/S3 (mean reversion)
                if close[i] >= camarilla_h3_aligned[i] and close[i] < camarilla_h4_aligned[i]:
                    # Short at R3 resistance
                    position = -1
                    signals[i] = -0.25
                elif close[i] <= camarilla_l3_aligned[i] and close[i] > camarilla_l4_aligned[i]:
                    # Long at S3 support
                    position = 1
                    signals[i] = 0.25
                # Breakout continuation at R4/S4 (trend following)
                elif close[i] > camarilla_h4_aligned[i]:
                    # Long breakout above R4
                    position = 1
                    signals[i] = 0.25
                elif close[i] < camarilla_l4_aligned[i]:
                    # Short breakdown below S4
                    position = -1
                    signals[i] = -0.25
    
    return signals