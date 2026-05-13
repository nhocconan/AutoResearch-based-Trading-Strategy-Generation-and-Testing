# 165085
#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Trend_1dEMA34
Hypothesis: Donchian(20) breakout on 12h with volume confirmation and 1d EMA34 trend filter.
Works in bull markets via breakout continuation and in bear via mean-reversion failed breakouts (quick exits).
Targets 20-50 trades/year to minimize fee drag. Uses 25% position size for risk control.
"""

name = "12h_Donchian_Breakout_Volume_Trend_1dEMA34"
timeframe = "12h"
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
    
    # Get 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend filter: EMA(34) on close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channels (20-period) on 12h data
    # Upper = max(high, lookback 20)
    # Lower = min(low, lookback 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above Donchian high, volume confirmation, price above 1d EMA34 (uptrend)
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low, volume confirmation, price below 1d EMA34 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian high (failed breakout) OR trend changes
            if (close[i] < donchian_high[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian low (failed breakdown) OR trend changes
            if (close[i] > donchian_low[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals