#!/usr/bin/env python3
# Hypothesis: 1d Donchian channel breakout with 1w trend filter (EMA34) and volume confirmation.
# Enters long when price breaks above upper Donchian(20) with price > 1w EMA34 and volume > MA20.
# Enters short when price breaks below lower Donchian(20) with price < 1w EMA34 and volume > MA20.
# Exits when price crosses the 1w EMA34 in the opposite direction.
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~10-20/year) to work in both bull and bear markets.

name = "1d_DonchianBreakout_1wTrend"
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Donchian channels (20-period) on 1d
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian with uptrend (price > 1w EMA34) and volume
            if close[i] > donchian_upper[i] and close[i] > ema34_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian with downtrend (price < 1w EMA34) and volume
            elif close[i] < donchian_lower[i] and close[i] < ema34_1w_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1w EMA34 (trend change)
            if close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1w EMA34 (trend change)
            if close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals