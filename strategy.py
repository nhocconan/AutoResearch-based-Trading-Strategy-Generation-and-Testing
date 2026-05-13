#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 12h Donchian upper (20) AND close > 1d EMA50 AND volume > 1.5x average
# Short when price breaks below 12h Donchian lower (20) AND close < 1d EMA50 AND volume > 1.5x average
# Exit when price crosses 12h Donchian middle (mean reversion) OR trend reversal (price crosses 1d EMA50)
# Uses 12h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with 1d trend filter for BTC/ETH resilience.
# Donchian provides clear structure; 1d EMA50 filters trend; volume spike confirms breakout authenticity.

name = "12h_Donchian20_1dEMA50_Volume_v2"
timeframe = "12h"
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
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period) - use rolling window on primary timeframe
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_ma = (high_ma + low_ma) / 2.0
    
    # Get 1d data for EMA50 trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 12h volume > 1.5x 20-period average (spike confirmation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA and Donchian
        # Skip if any required data is NaN
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or np.isnan(mid_ma[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Donchian upper AND close > 1d EMA50 AND volume spike
            if close[i] > high_ma[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < Donchian lower AND close < 1d EMA50 AND volume spike
            elif close[i] < low_ma[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Donchian middle (mean reversion) OR trend reversal (close < 1d EMA50)
            if close[i] < mid_ma[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > Donchian middle (mean reversion) OR trend reversal (close > 1d EMA50)
            if close[i] > mid_ma[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals