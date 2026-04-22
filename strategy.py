#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla R1/S1 breakout with 12-hour EMA trend filter and volume confirmation.
Long when price breaks above R1 with 12h EMA50 uptrend and volume > 1.5x average.
Short when price breaks below S1 with 12h EMA50 downtrend and volume > 1.5x average.
Exit when price reaches opposite Camarilla level (S1 for longs, R1 for shorts) or reverses trend.
Camarilla levels provide intraday support/resistance; EMA50 filters trend direction; volume ensures conviction.
Works in bull markets via breakouts and in bear markets via mean reversion at extreme levels.
"""

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
    
    # Load 1-day data for Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with uptrend and volume confirmation
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i-1] <= camarilla_r1_aligned[i-1] and
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with downtrend and volume confirmation
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i-1] >= camarilla_s1_aligned[i-1] and
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches S1 or trend turns down
                if (close[i] <= camarilla_s1_aligned[i] or 
                    ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price reaches R1 or trend turns up
                if (close[i] >= camarilla_r1_aligned[i] or 
                    ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0