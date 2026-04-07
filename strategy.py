#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v3
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) upper band with volume confirmation in uptrend (price > 1d EMA50).
Enter short when price breaks below Donchian(20) lower band with volume confirmation in downtrend (price < 1d EMA50).
Exit on opposite Donchian break or trend reversal. Uses Donchian breakouts for trend continuation, volume for confirmation,
and EMA50 for trend filter to avoid counter-trend trades. Designed for 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v3"
timeframe = "4h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # Donchian channels (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume in uptrend
            if (close[i] > donchian_high[i] and
                vol_confirm and 
                close[i] > ema_50_aligned[i]):  # uptrend filter
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume in downtrend
            elif (close[i] < donchian_low[i] and
                  vol_confirm and 
                  close[i] < ema_50_aligned[i]):  # downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals