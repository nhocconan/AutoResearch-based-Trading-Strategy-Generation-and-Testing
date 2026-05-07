#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R3 level from previous day, 
and short when price breaks below S3 level, with confirmation from 1d EMA trend and volume spikes.
Camarilla levels provide precise intraday support/resistance; EMA trend ensures alignment with higher timeframe direction;
volume filter avoids false breakouts. Designed for fewer trades (12-37/year) to minimize fee drag.
Works in bull markets via breakouts above R3 and in bear markets via breakdowns below S3.
"""
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous day (using OHLC of prior day)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We use R3 and S3 as breakout levels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla R3 and S3 for each day
    H_L = high_1d - low_1d
    C = close_1d
    R3 = C + H_L * 1.1 / 4
    S3 = C - H_L * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (values from previous day)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = 1  # start from second bar to ensure we have previous day's levels
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 4 bars between trades to reduce frequency (12h timeframe)
            if bars_since_exit < 4:
                continue
                
            # Long: price breaks above R3 + price above EMA34 + volume filter
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S3 + price below EMA34 + volume filter
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level (S3 for long, R3 for short)
            if position == 1:
                if close[i] < S3_aligned[i]:  # price breaks below S3
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > R3_aligned[i]:  # price breaks above R3
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = -0.25
    
    return signals