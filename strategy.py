# US-116218: 4h Camarilla breakout with volume confirmation - optimized for lower trade count
# Uses 1d Camarilla H3/L3 levels with 1d volume > 1.5x 20-day average to reduce false signals
# Exit on close crossing H3/L3 level or reversal signal
# Target: 30-50 trades/year to stay under 400 total over 4 years

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v28"
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar data (avoid look-ahead)
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Calculate 1d Camarilla H3/L3 levels (breakout levels)
    pivot_prev = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d_prev = high_1d_prev - low_1d_prev
    h3_prev = pivot_prev + (range_1d_prev * 1.1 / 4)
    l3_prev = pivot_prev - (range_1d_prev * 1.1 / 4)
    
    # Align to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_prev)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_prev)
    
    # 1d volume spike: volume > 1.5x 20-day average (reduced threshold for better signal quality)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'] > (vol_ma_1d * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above H3 with volume spike
        long_signal = close[i] > h3_aligned[i] and vol_spike_aligned[i]
        # Short: break below L3 with volume spike
        short_signal = close[i] < l3_aligned[i] and vol_spike_aligned[i]
        
        # Exit when price crosses H3/L3 level (mean reversion)
        exit_long = close[i] < h3_aligned[i]
        exit_short = close[i] > l3_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals