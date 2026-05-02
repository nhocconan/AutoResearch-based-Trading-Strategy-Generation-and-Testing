#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend filter and 12h Donchian channels for breakout signals
# Entry: Long when price breaks above upper Donchian(20) AND price > 1d EMA34 AND volume spike
#        Short when price breaks below lower Donchian(20) AND price < 1d EMA34 AND volume spike
# Exit: Price crosses the 12h EMA20 (mid-channel) OR Donchian breakout in opposite direction
# Works in both bull and bear markets by trading with 1d trend using 12h Donchian breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "12h_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian(20) channels
    period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window_high = high[i - period + 1:i + 1]
        window_low = low[i - period + 1:i + 1]
        upper[i] = np.max(window_high)
        lower[i] = np.min(window_low)
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(100, period)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(middle[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above upper Donchian AND price > 1d EMA34 AND volume spike
            if (close[i] > upper[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below lower Donchian AND price < 1d EMA34 AND volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price crosses below middle Donchian (mean reversion) OR Donchian breakdown
            if close[i] < middle[i] or close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price crosses above middle Donchian (mean reversion) OR Donchian breakout
            if close[i] > middle[i] or close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals