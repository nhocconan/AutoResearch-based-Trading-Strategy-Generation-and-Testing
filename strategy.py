# Forced 1h version of 4h_1d_camarilla_breakout_v25 - proven winner
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 1d Camarilla pivot levels + volume confirmation
# Long when price breaks above H3 with volume > 1.5x MA(20)
# Short when price breaks below L3 with volume > 1.5x MA(20)
# Exit when price crosses opposite pivot level (H3->L3 or L3->H3)
# Uses 1d for structure to avoid whipsaws, 1h only for entry timing
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # Using (H-L) from previous day
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses current day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_hl = prev_high - prev_low
    h3 = prev_close + 1.1 * range_hl / 2
    l3 = prev_close - 1.1 * range_hl / 2
    h4 = prev_close + 1.1 * range_hl
    l4 = prev_close - 1.1 * range_hl
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Align 1d levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: break of H3/L3 with volume
        long_break = close[i] > h3_aligned[i] and vol_filter_aligned[i]
        short_break = close[i] < l3_aligned[i] and vol_filter_aligned[i]
        
        # Exit conditions: cross opposite pivot level
        exit_long = position == 1 and close[i] < l3_aligned[i]
        exit_short = position == -1 and close[i] > h3_aligned[i]
        
        # Execute signals
        if long_break and position != 1:
            position = 1
            signals[i] = position_size
        elif short_break and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_1d_camarilla_breakout_v25"
timeframe = "1h"
leverage = 1.0