# 165117
#!/usr/bin/env python3
"""
4h_Donchian_20_Upper_Lower_Breakout_VolumeTrend
Hypothesis: Donchian channel (20-period) breakouts on 4h timeframe, confirmed by volume and trend alignment, provide strong directional moves. Uses 20-period Donchian for upper/lower bands, volume > 1.5x 20-period average, and price > 50-period EMA for trend filter. Designed for moderate trade frequency (~25-40/year) to balance signal quality and fee drag in 4-hour bars. Works in both bull and bear markets by capturing breakouts in either direction with proper filters.
"""

name = "4h_Donchian_20_Upper_Lower_Breakout_VolumeTrend"
timeframe = "4h"
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
    
    # Calculate 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Trend filter: 50-period EMA on close
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above Donchian upper band, volume confirmation, price above EMA50 (uptrend)
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band, volume confirmation, price below EMA50 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian upper band (failed breakout) OR trend reverses
            if (close[i] < donchian_high[i] or 
                close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian lower band (failed breakdown) OR trend reverses
            if (close[i] > donchian_low[i] or 
                close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals