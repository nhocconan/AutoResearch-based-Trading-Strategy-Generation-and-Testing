#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA(50) trend filter and volume confirmation (1.8x 20-period average)
# Long when price breaks above Donchian upper, price > 12h EMA(50), and volume > 1.8x average
# Short when price breaks below Donchian lower, price < 12h EMA(50), and volume > 1.8x average
# Exit when price crosses opposite Donchian band or trend reverses (price crosses EMA)
# Position size: 0.28 (28% of capital)
# Uses 12h trend to filter false breakouts and align with higher timeframe bias
# Target: 100-180 total trades over 4 years (25-45/year)

name = "4h_donchian20_12h_ema50_vol_v1"
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
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.28
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian lower or trend turns bearish
            if close[i] < donchian_lower[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.28
        elif position == -1:  # short position
            # Exit: price crosses above Donchian upper or trend turns bullish
            if close[i] > donchian_upper[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.28
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above Donchian upper, price above EMA (bullish trend), volume confirmation
            if (close[i] > donchian_upper[i] and
                close[i] > ema_12h_aligned[i] and
                volume[i] > 1.8 * volume_ma[i]):
                signals[i] = 0.28
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower, price below EMA (bearish trend), volume confirmation
            elif (close[i] < donchian_lower[i] and
                  close[i] < ema_12h_aligned[i] and
                  volume[i] > 1.8 * volume_ma[i]):
                signals[i] = -0.28
                position = -1
                entry_price = close[i]
    
    return signals