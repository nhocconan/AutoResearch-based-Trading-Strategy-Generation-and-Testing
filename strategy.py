#!/usr/bin/env python3
"""
1h_sr_breakout_4h1d_trend_volume_v1
Hypothesis: On 1h timeframe, trade breakouts of 4h support/resistance levels with 1d trend filter and volume confirmation. Enter long when price breaks above 4h resistance with 1d EMA50 > EMA200 and volume > 1.5x average; enter short when price breaks below 4h support with 1d EMA50 < EMA200 and volume > 1.5x average. Exit when price returns to 4h midpoint or trend reverses. Uses 4h for structure, 1d for trend filter, 1h for entry timing. Targets 15-35 trades/year to avoid fee drag. Works in bull/bear via 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_sr_breakout_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    ema50_1d = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 4h support/resistance levels (using prior 4h bar's high/low/close)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ph_4h = df_4h['high'].values  # previous 4h high
    pl_4h = df_4h['low'].values   # previous 4h low
    pc_4h = df_4h['close'].values # previous 4h close
    
    # Calculate 4h pivot and ranges
    pivot_4h = (ph_4h + pl_4h + pc_4h) / 3
    range_4h = ph_4h - pl_4h
    
    # 4h resistance (R1) and support (S1) levels
    r1_4h = pivot_4h + range_4h
    s1_4h = pivot_4h - range_4h
    # Midpoint for exit
    mid_4h = pivot_4h
    
    # Align to 1h timeframe (shifted by 1 bar for look-ahead prevention)
    r1_4h_1h = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_1h = align_htf_to_ltf(prices, df_4h, s1_4h)
    mid_4h_1h = align_htf_to_ltf(prices, df_4h, mid_4h)
    
    # Volume confirmation (24-period average on 1h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema50_1d[i]) or np.isnan(ema200_1d[i]) or 
            np.isnan(r1_4h_1h[i]) or np.isnan(s1_4h_1h[i]) or
            np.isnan(mid_4h_1h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price returns to 4h midpoint
            if close[i] <= mid_4h_1h[i]:
                exit_long = True
            # Exit if 1d EMA50 crosses below EMA200 (trend reversal)
            elif ema50_1d[i] < ema200_1d[i] and ema50_1d[i-1] >= ema200_1d[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price returns to 4h midpoint
            if close[i] >= mid_4h_1h[i]:
                exit_short = True
            # Exit if 1d EMA50 crosses above EMA200 (trend reversal)
            elif ema50_1d[i] > ema200_1d[i] and ema50_1d[i-1] <= ema200_1d[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: price breaks above 4h resistance with 1d uptrend and volume confirmation
            long_entry = False
            if (close[i] > r1_4h_1h[i] and close[i-1] <= r1_4h_1h[i-1] and
                ema50_1d[i] > ema200_1d[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below 4h support with 1d downtrend and volume confirmation
            short_entry = False
            if (close[i] < s1_4h_1h[i] and close[i-1] >= s1_4h_1h[i-1] and
                ema50_1d[i] < ema200_1d[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.20
            elif short_entry:
                position = -1
                signals[i] = -0.20
    
    return signals