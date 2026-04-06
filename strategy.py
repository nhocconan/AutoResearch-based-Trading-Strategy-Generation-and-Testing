#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter, volume confirmation, and ATR stoploss
# Long when price breaks above Donchian(20) high + 1d EMA(50) uptrend + volume > 20-period average
# Short when price breaks below Donchian(20) low + 1d EMA(50) downtrend + volume > 20-period average
# Exit when price crosses Donchian midline or trend reverses
# Target: 75-200 total trades over 4 years with controlled risk in both bull and bear markets

name = "4h_donchian20_1d_ema_vol_v1"
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
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2.0
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian midline or 1d EMA turns down
            elif close[i] < donchian_mid[i] or ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian midline or 1d EMA turns up
            elif close[i] > donchian_mid[i] or ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_filter[i]:
                # Long when price breaks above Donchian high + 1d EMA uptrend
                if close[i] > high_max[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when price breaks below Donchian low + 1d EMA downtrend
                elif close[i] < low_min[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals