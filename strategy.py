#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels derived from the daily (1D) range provide strong intraday support/resistance levels. 
A breakout above R1 or below S1, aligned with the 1D trend (EMA34), and confirmed by volume spikes, captures directional momentum. 
This strategy avoids overtrading by requiring multiple confluence factors and uses the 4H timeframe for lower trade frequency.
The 1D EMA34 trend filter ensures alignment with the higher timeframe direction, reducing false signals in choppy markets.
Volume confirmation adds conviction to breakouts.
Target: 20-50 trades per year (80-200 over 4 years).
"""
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    # Camarilla: R1 = close + (high - low) * 1.1 / 12, S1 = close - (high - low) * 1.1 / 12
    # We use the previous day's OHLC to avoid look-ahead
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate R1 and S1 for each day
    r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align these levels to the 4H timeframe (they are valid for the entire day after the daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1D EMA34 for trend filter (using daily close)
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average (on 4H)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 1D uptrend + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 1D downtrend + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to the opposite Camarilla level or crosses the EMA
            if position == 1:
                # Exit long if price breaks below S1 or crosses below EMA
                if close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price breaks above R1 or crosses above EMA
                if close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals