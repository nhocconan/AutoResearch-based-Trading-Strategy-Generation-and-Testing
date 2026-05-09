#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrendFilter_Volume
# Strategy: Camarilla pivot breakout on 1d with 1w trend filter and volume confirmation
# Long when price breaks above R1 and close > 1w EMA50 and volume > 1.5x avg volume
# Short when price breaks below S1 and close < 1w EMA50 and volume > 1.5x avg volume
# Exit when price returns to Camarilla pivot point (PP)
# Designed for 1d timeframe with selective entries to minimize trade frequency and avoid overtrading

name = "1d_Camarilla_R1_S1_Breakout_1wTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Calculate 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # Camarilla pivot point
            pp = (prev_high + prev_low + prev_close) / 3
            # R1 and S1 levels
            r1 = pp + (prev_high - prev_low) * 1.0833 / 12
            s1 = pp - (prev_high - prev_low) * 1.0833 / 12
            
            # Volume confirmation: current volume > 1.5x average volume
            volume_confirmed = volume[i] > 1.5 * avg_volume[i]
            
            if position == 0:
                # Enter long: price breaks above R1 with trend and volume confirmation
                if close[i] > r1 and close[i] > ema_50_1w_aligned[i] and volume_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Enter short: price breaks below S1 with trend and volume confirmation
                elif close[i] < s1 and close[i] < ema_50_1w_aligned[i] and volume_confirmed:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Exit long: price returns to pivot point
                if close[i] <= pp:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                # Exit short: price returns to pivot point
                if close[i] >= pp:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals