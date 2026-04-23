#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above Donchian(20) upper band AND 12h EMA50 is rising AND volume > 1.5x average.
Short when price breaks below Donchian(20) lower band AND 12h EMA50 is falling AND volume > 1.5x average.
Exit when price crosses opposite Donchian band or volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Donchian breakouts capture strong momentum moves, EMA filter ensures trend alignment, volume confirms conviction.
Works in both bull and bear markets via directional filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA slope for trend direction
        if i >= 1:
            ema_slope = ema50_12h_aligned[i] - ema50_12h_aligned[i-1]
        else:
            ema_slope = 0
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above upper band AND rising EMA AND volume confirmation
            if (close[i] > highest_high[i] and ema_slope > 0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND falling EMA AND volume confirmation
            elif (close[i] < lowest_low[i] and ema_slope < 0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below lower band OR volume drops below average
                if (close[i] < lowest_low[i] or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above upper band OR volume drops below average
                if (close[i] > highest_high[i] or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0