# Hypothesis: 4h 12h/1d pivot breakout with volume confirmation and 12h EMA trend filter
# Uses 12h/1d pivot levels as strong support/resistance zones. Long when price breaks above R1 with volume and above 12h EMA34.
# Short when price breaks below S1 with volume and below 12h EMA34. Designed for fewer trades (target 20-50/year) with clear structure.
# Works in bull/bear: pivot levels adapt to volatility, EMA filter avoids counter-trend trades, volume confirms breakout strength.
# Target: 0.25 position size, max 0.40.

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
    
    # Get 12h data for pivot calculation (more stable than daily)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h pivot points (standard formula)
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    r2_12h = pivot_12h + range_12h
    s2_12h = pivot_12h - range_12h
    
    # Align 12h pivot levels to 4h
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    
    # Get 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume filter: current volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or
            np.isnan(r2_12h_aligned[i]) or np.isnan(s2_12h_aligned[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and above 12h EMA34
            if close[i] > r1_12h_aligned[i] and volume_filter[i] and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below 12h EMA34
            elif close[i] < s1_12h_aligned[i] and volume_filter[i] and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below pivot OR below S1
            if close[i] < pivot_12h_aligned[i] or close[i] < s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above pivot OR above R1
            if close[i] > pivot_12h_aligned[i] or close[i] > r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hPivot_R1S1_Volume_EMA34"
timeframe = "4h"
leverage = 1.0