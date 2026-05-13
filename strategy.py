#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND close > 1w EMA34 AND volume > 1.5x 20-period average volume.
# Short when price breaks below Donchian(20) low AND close < 1w EMA34 AND volume > 1.5x 20-period average volume.
# Exit when price touches the opposite Donchian(20) level (low for long exit, high for short exit).
# Uses discrete position sizes (0.0, ±0.25) to minimize fee churn. Designed for 7-25 trades/year by requiring
# strong breakouts with volume confirmation and weekly trend alignment. Works in bull markets via breakout momentum
# and in bear markets via breakdowns with weekly trend filter to avoid counter-trend whipsaws.

name = "1d_Donchian20_1wTrend_Volume_v1"
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
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average volume(20) for volume confirmation
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = avg_vol_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Donchian
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(avg_vol_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian(20) high AND close > 1w EMA34 AND volume > 1.5x average volume
            if close[i] > highest_high[i] and close[i] > ema34_1w_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian(20) low AND close < 1w EMA34 AND volume > 1.5x average volume
            elif close[i] < lowest_low[i] and close[i] < ema34_1w_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches Donchian(20) low
            if close[i] <= lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches Donchian(20) high
            if close[i] >= highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals