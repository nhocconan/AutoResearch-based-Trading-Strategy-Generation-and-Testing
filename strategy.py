#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour 1-week Camarilla pivot breakout with volume confirmation.
# Uses weekly Camarilla levels (H5, L5, H6, L6) calculated from previous week's range.
# Long when price breaks above H6 with volume > 1.5x 20-period average.
# Short when price breaks below L6 with volume > 1.5x 20-period average.
# Exit when price returns to weekly midpoint (H4/L4 average).
# Weekly Camarilla provides institutional support/resistance, breakouts capture momentum,
# volume confirms institutional participation. Works in trending markets (bull/bear).
name = "6h_WeeklyCamarilla_H6L6_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla levels from previous week's data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's high, low, close
    prev_week_high = df_1w['high'].shift(1).values  # Previous week's high
    prev_week_low = df_1w['low'].shift(1).values    # Previous week's low
    prev_week_close = df_1w['close'].shift(1).values # Previous week's close
    
    # Calculate Camarilla levels for current week based on previous week
    # H6 = Close + 1.163*(High - Low)
    # L6 = Close - 1.163*(High - Low)
    # H4 = Close + 0.550*(High - Low)
    # L4 = Close - 0.550*(High - Low)
    range_wk = prev_week_high - prev_week_low
    h6 = prev_week_close + 1.163 * range_wk
    l6 = prev_week_close - 1.163 * range_wk
    h4 = prev_week_close + 0.550 * range_wk
    l4 = prev_week_close - 0.550 * range_wk
    midpoint = (h4 + l4) / 2  # Weekly midpoint for exit
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    h6_aligned = align_htf_to_ltf(prices, df_1w, h6)
    l6_aligned = align_htf_to_ltf(prices, df_1w, l6)
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h6_aligned[i]) or np.isnan(l6_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above H6 with volume spike
            if price > h6_aligned[i] and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below L6 with volume spike
            elif price < l6_aligned[i] and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly midpoint
            if price <= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly midpoint
            if price >= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals