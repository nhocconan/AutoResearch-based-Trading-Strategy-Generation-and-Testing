#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_v1
# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation.
# Long: Close above H3 pivot level + volume > 1.5x 20-period average.
# Short: Close below L3 pivot level + volume > 1.5x 20-period average.
# Exit: Opposite pivot level touch (H4/L4) or volume divergence.
# Uses 1d pivot levels calculated from prior 1d OHLC for structure.
# Volume confirmation filters weak breakouts. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_v1"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = Pivot + Range * 1.1/2
    # H3 = Pivot + Range * 1.1/4
    # L3 = Pivot - Range * 1.1/4
    # L4 = Pivot - Range * 1.1/2
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot = typical_price.values
    df_range = df_1d['high'] - df_1d['low']
    range_val = df_range.values
    
    h4 = pivot + range_val * 1.1 / 2
    h3 = pivot + range_val * 1.1 / 4
    l3 = pivot - range_val * 1.1 / 4
    l4 = pivot - range_val * 1.1 / 2
    
    # Align 1d pivot levels to 12h timeframe (wait for completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(l4_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Close below L3 (mean reversion) OR volume divergence (price up but volume down)
            if close[i] < l3_aligned[i] or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Close above H3 (mean reversion) OR volume divergence (price down but volume down)
            if close[i] > h3_aligned[i] or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Close above H3 with volume confirmation
            if (close[i] > h3_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Close below L3 with volume confirmation
            elif (close[i] < l3_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals