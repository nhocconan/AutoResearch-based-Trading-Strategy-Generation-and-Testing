#!/usr/bin/env python3
"""
6h_donchian_breakout_12h_trend_volume_v1
Hypothesis: On 6h timeframe, trade Donchian channel breakouts filtered by 12h trend (EMA) and volume confirmation. 
In uptrend (price > 12h EMA200), go long on breakout above 20-period high. 
In downtrend (price < 12h EMA200), go short on breakout below 20-period low. 
Volume confirms genuine breakouts (not fakeouts). This captures trends in both bull and bear markets while avoiding chop.
Target: 12-37 trades/year (~50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA200 for trend filter
    ema_200 = df_12h['close'].ewm(span=200, adjust=False).mean()
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200.values)
    
    # Donchian channels (20-period) on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (24-period average on 6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume in uptrend
            if (close[i] > donchian_high[i] and
                vol_confirm and 
                close[i] > ema_200_aligned[i]):  # uptrend filter
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume in downtrend
            elif (close[i] < donchian_low[i] and
                  vol_confirm and 
                  close[i] < ema_200_aligned[i]):  # downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals