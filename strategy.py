# 165118
#!/usr/bin/env python3
"""
1d_Donchian_20_Weekly_Trend_Volume
Hypothesis: Daily Donchian(20) breakouts with weekly trend alignment and volume confirmation work in both bull and bear markets. Weekly trend filter avoids counter-trend trades, reducing whipsaws. Volume confirmation ensures breakout strength. Designed for low trade frequency (~10-25/year) to minimize fee drag on daily bars.
"""

name = "1d_Donchian_20_Weekly_Trend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Daily Donchian channels (20-period)
    # Upper = max(high, 20)
    # Lower = min(low, 20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: EMA(34) on weekly close
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above Donchian upper, weekly uptrend, volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower, weekly downtrend, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian lower (failed breakout) OR weekly trend turns down
            if (close[i] < donchian_low[i] or 
                close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian upper (failed breakdown) OR weekly trend turns up
            if (close[i] > donchian_high[i] or 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals