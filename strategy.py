#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1w EMA34 is rising AND volume > 1.5x 20-bar average volume.
# Short when price breaks below Donchian(20) low AND 1w EMA34 is falling AND volume > 1.5x 20-bar average volume.
# Exit when price touches the opposite Donchian(20) level (low for long exit, high for short exit).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.
# Designed for 7-25 trades/year on 1d timeframe by requiring strong breakouts with volume and trend confirmation.

name = "1d_Donchian20_1wTrend_Volume_v2"
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Donchian channels (20-bar)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Donchian
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or \
           np.isnan(avg_vol_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND 1w EMA34 rising AND volume confirmation
            if close[i] > highest_20[i] and ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND 1w EMA34 falling AND volume confirmation
            elif close[i] < lowest_20[i] and ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches Donchian low (mean reversion exit)
            if close[i] <= lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches Donchian high (mean reversion exit)
            if close[i] >= highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals