# 12h_pivot_breakout_volume_v2
# Hypothesis: Combines 12h price action at Camarilla pivot levels with volume confirmation and weekly trend filter.
# Long when: price breaks above R4 level with volume > 2x average and weekly trend up.
# Short when: price breaks below S4 level with volume > 2x average and weekly trend down.
# Exit when price returns to Pivot point or volume drops below average.
# Uses Camarilla pivots from daily timeframe for institutional levels, weekly trend for direction filter.
# Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_pivot_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 2x 24-period average (2 days)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 2.0 * vol_ma[i]
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (H-L) * 1.1/2
    # S4 = C - (H-L) * 1.1/2
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align daily pivot levels to 12h timeframe (previous day's levels)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Get weekly data for trend filter (SMA50 slope)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma_period = 50
    sma50_1w = pd.Series(close_1w).rolling(window=sma_period, min_periods=sma_period).mean().values
    # Calculate slope: positive if current SMA > SMA 2 periods ago
    sma50_slope_1w = np.full(len(close_1w), np.nan)
    for i in range(2, len(close_1w)):
        if not np.isnan(sma50_1w[i]) and not np.isnan(sma50_1w[i-2]):
            sma50_slope_1w[i] = sma50_1w[i] - sma50_1w[i-2]
    sma50_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_slope_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 2) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma[i]) or np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(sma50_slope_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot point or volume drops below average
            if close[i] <= pp_1d_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot point or volume drops below average
            if close[i] >= pp_1d_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above R4 with volume surge and weekly trend up
            if (close[i] > r4_1d_aligned[i] and 
                vol_surge[i] and 
                sma50_slope_1w_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S4 with volume surge and weekly trend down
            elif (close[i] < s4_1d_aligned[i] and 
                  vol_surge[i] and 
                  sma50_slope_1w_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals