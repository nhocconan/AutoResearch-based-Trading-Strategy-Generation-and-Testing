#!/usr/bin/env python3
# 12h_daily_pivot_volume_filter_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation.
# Long: price crosses above H3 (bullish breakout) with volume > 1.5x 20-period average
# Short: price crosses below L3 (bearish breakdown) with volume > 1.5x 20-period average
# Exit: price returns to P (pivot point) or opposite pivot level touched
# Designed to capture institutional breakout/breakdown moves with volume validation.
# Works in both bull/bear markets as it captures momentum shifts regardless of direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_pivot_volume_filter_v1"
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
    
    # Get daily data for Camarilla pivots (calculated from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous daily bar
    # H4, H3, H2, H1, L1, L2, L3, L4
    # Pivot = (H + L + C) / 3
    # H3 = P + (H - L) * 1.1 / 4
    # L3 = P - (H - L) * 1.1 / 4
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    rang = prev_high - prev_low
    h3 = pivot + rang * 1.1 / 4
    l3 = pivot - rang * 1.1 / 4
    
    # Align to 12h timeframe (using previous day's pivots)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / 20
            vol_sum -= volume[i - 19]
            vol_count -= 1
    
    volume_filter = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            volume_filter[i] = volume[i] > (vol_ma[i] * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any values are NaN
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pivot_aligned[i]):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        vol_ok = volume_filter[i]
        price = close[i]
        
        if position == 1:  # Long position
            # Exit: price returns to pivot or breaks below L3
            if price <= pivot_aligned[i] or price < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot or breaks above H3
            if price >= pivot_aligned[i] or price > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volume confirmation
            if price > h3_aligned[i] and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume confirmation
            elif price < l3_aligned[i] and vol_ok:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
# 12h_daily_pivot_volume_filter_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation.
# Long: price crosses above H3 (bullish breakout) with volume > 1.5x 20-period average
# Short: price crosses below L3 (bearish breakdown) with volume > 1.5x 20-period average
# Exit: price returns to P (pivot point) or opposite pivot level touched
# Designed to capture institutional breakout/breakdown moves with volume validation.
# Works in both bull/bear markets as it captures momentum shifts regardless of direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_pivot_volume_filter_v1"
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
    
    # Get daily data for Camarilla pivots (calculated from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous daily bar
    # H4, H3, H2, H1, L1, L2, L3, L4
    # Pivot = (H + L + C) / 3
    # H3 = P + (H - L) * 1.1 / 4
    # L3 = P - (H - L) * 1.1 / 4
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    rang = prev_high - prev_low
    h3 = pivot + rang * 1.1 / 4
    l3 = pivot - rang * 1.1 / 4
    
    # Align to 12h timeframe (using previous day's pivots)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / 20
            vol_sum -= volume[i - 19]
            vol_count -= 1
    
    volume_filter = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            volume_filter[i] = volume[i] > (vol_ma[i] * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any values are NaN
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pivot_aligned[i]):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        vol_ok = volume_filter[i]
        price = close[i]
        
        if position == 1:  # Long position
            # Exit: price returns to pivot or breaks below L3
            if price <= pivot_aligned[i] or price < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot or breaks above H3
            if price >= pivot_aligned[i] or price > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volume confirmation
            if price > h3_aligned[i] and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume confirmation
            elif price < l3_aligned[i] and vol_ok:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals