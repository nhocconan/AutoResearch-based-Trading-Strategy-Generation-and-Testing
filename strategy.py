#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band, price > 1d EMA50, and volume > 1.5x volume MA(20).
# Short when price breaks below Donchian lower band, price < 1d EMA50, and volume > 1.5x volume MA(20).
# Exit when price crosses back below Donchian middle band (for longs) or above (for shorts).
# Uses volatility-adjusted position sizing (0.25) to manage drawdowns.
# Target: 20-50 trades per year per symbol (80-200 over 4 years).

name = "4h_Donchian20_EMA50_Volume_Trend"
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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: breakout above Donchian high, above EMA50, volume confirmed
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: breakdown below Donchian low, below EMA50, volume confirmed
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian middle band
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian middle band
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals