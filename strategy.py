#!/usr/bin/env python3
# 1h_4h1d_donchian_volume_trend
# Hypothesis: 1-hour trend-following strategy using 4h and 1d Donchian channels for direction and 1h volume breakouts for entry.
# In bull markets: 4h Donchian upper break + 1d uptrend + 1h volume spike = long
# In bear markets: 4h Donchian lower break + 1d downtrend + 1h volume spike = short
# Uses Donchian channels (20-period) on 4h for trend, 1d EMA200 for higher timeframe bias, and volume confirmation on 1h.
# Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_donchian_volume_trend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period) - load once
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian upper/lower
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # 1d trend filter (EMA200) - load once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h volume average (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # 1d trend bias
        bullish_bias = close[i] > ema200_1d_aligned[i]
        bearish_bias = close[i] < ema200_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: close below 4h Donchian lower or trend reversal
            if close[i] < donch_low_4h_aligned[i] or (bearish_bias and close[i] < ema200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: close above 4h Donchian upper or trend reversal
            if close[i] > donch_high_4h_aligned[i] or (bullish_bias and close[i] > ema200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: break above 4h Donchian upper in bullish bias
                if bullish_bias and close[i] > donch_high_4h_aligned[i] and close[i-1] <= donch_high_4h_aligned[i-1]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: break below 4h Donchian lower in bearish bias
                elif bearish_bias and close[i] < donch_low_4h_aligned[i] and close[i-1] >= donch_low_4h_aligned[i-1]:
                    position = -1
                    signals[i] = -0.20
    
    return signals