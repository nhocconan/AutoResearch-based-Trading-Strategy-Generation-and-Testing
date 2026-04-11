# #!/usr/bin/env python3
# 12h_1d_donchian_breakout_v1
# Strategy: 12h Donchian breakout with 1d trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture trend continuations. Combined with 1d trend filter (price > SMA50) and volume spike, this reduces false signals. Works in bull by buying breakouts above resistance, works in bear by selling breakdowns below support. Volume confirmation ensures breakouts have conviction. Designed for low trade frequency to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channels on 12h (20-period)
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # 1d SMA50 for trend filter
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean()
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(sma_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price relative to 1d SMA50
        price_above_sma50 = close[i] > sma_50_1d_aligned[i]
        price_below_sma50 = close[i] < sma_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above Donchian upper AND above 1d SMA50 AND volume spike
        if (close[i] > donchian_upper[i] and price_above_sma50 and volume_spike[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below Donchian lower AND below 1d SMA50 AND volume spike
        elif (close[i] < donchian_lower[i] and price_below_sma50 and volume_spike[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses opposite Donchian band or trend filter fails
        elif position == 1 and (close[i] < donchian_lower[i] or not price_above_sma50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_upper[i] or not price_below_sma50):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals