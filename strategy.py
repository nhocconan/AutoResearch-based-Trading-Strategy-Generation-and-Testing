#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
Long when price breaks above R3 (1d) AND 1d close > 1d EMA34 (uptrend) AND volume > 1.5x 20-period MA.
Short when price breaks below S3 (1d) AND 1d close < 1d EMA34 (downtrend) AND volume > 1.5x 20-period MA.
Exit when price retouches the Camarilla pivot (close crosses P) or 1d trend reverses.
Camarilla levels provide precise intraday support/resistance; 1d EMA34 filters counter-trend trades; volume confirms breakout strength.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

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
    
    # Calculate 1d Camarilla levels (R3, S3, P)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    # P = (high + low + close)/3
    rng = high_1d - low_1d
    R3 = close_1d + 1.1 * rng / 2
    S3 = close_1d - 1.1 * rng / 2
    P = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(P_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d close > EMA34 = uptrend, close < EMA34 = downtrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend AND volume filter
            if close[i] > R3_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND downtrend AND volume filter
            elif close[i] < S3_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price retouches pivot (close crosses below P) OR 1d trend turns down
                if close[i] < P_aligned[i] or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: price retouches pivot (close crosses above P) OR 1d trend turns up
                if close[i] > P_aligned[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeFilter"
timeframe = "12h"
leverage = 1.0