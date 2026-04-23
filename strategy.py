#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high AND 12h EMA50 > price (uptrend) AND volume > 1.5x 20-period average.
Short when price breaks below 20-period Donchian low AND 12h EMA50 < price (downtrend) AND volume > 1.5x 20-period average.
Exit when price touches the opposite Donchian level (Donchian low for longs, Donchian high for shorts).
Uses 12h HTF for EMA50 trend filter to avoid whipsaws in ranging markets. Target: 75-200 total trades over 4 years (19-50/year).
Donchian breakouts capture strong momentum moves; EMA50 filter ensures we only trade in the direction of the 12h trend.
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
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50 + 49)  # donchian (20), ema calculation (50+49 for ewm)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_12h_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above Donchian high AND 12h EMA50 > price (uptrend) AND volume spike
            if price > upper and ema_val > price and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND 12h EMA50 < price (downtrend) AND volume spike
            elif price < lower and ema_val < price and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Donchian level
            if position == 1 and price < lower:  # Long exit at Donchian low
                exit_signal = True
            elif position == -1 and price > upper:  # Short exit at Donchian high
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirmation_LevelExit"
timeframe = "4h"
leverage = 1.0