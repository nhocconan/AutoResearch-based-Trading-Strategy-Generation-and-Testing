#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.5x 20-period average volume.
# Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.5x 20-period average volume.
# Exit on opposite Donchian(10) break or when price crosses 1d EMA34.
# Position size: 0.25 (discrete level to minimize fee churn). Target: 20-50 trades/year.
# Works in bull markets via breakout momentum and in bear markets via trend-filtered short breakdowns.

name = "4h_Donchian20_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channels (20-period for entry, 10-period for exit)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average volume
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Donchian(20)
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or \
           np.isnan(highest_10[i]) or np.isnan(lowest_10[i]) or np.isnan(avg_vol_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian(20) high AND price > 1d EMA34 AND volume confirmation
            if close[i] > highest_20[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian(20) low AND price < 1d EMA34 AND volume confirmation
            elif close[i] < lowest_20[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian(10) low OR price crosses below 1d EMA34
            if close[i] < lowest_10[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian(10) high OR price crosses above 1d EMA34
            if close[i] > highest_10[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals