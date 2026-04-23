#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
- Long when price breaks above Donchian upper band (20-period high) AND price > 1d EMA34 AND volume > 1.5x 20-period average
- Short when price breaks below Donchian lower band (20-period low) AND price < 1d EMA34 AND volume > 1.5x 20-period average
- Exit when price returns to Donchian midpoint (10-period average of high/low) OR trend reverses
- Uses discrete position sizing (0.25) to minimize fee churn
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years)
- Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter
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
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > 1d EMA34 AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND price < 1d EMA34 AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Donchian midpoint OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price <= Donchian midpoint OR price closes below 1d EMA34
                if (close[i] <= donchian_mid[i] or close[i] < ema_34_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price >= Donchian midpoint OR price closes above 1d EMA34
                if (close[i] >= donchian_mid[i] or close[i] > ema_34_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0