#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 1d EMA filter
# Long when price breaks above Donchian high (20) on 4h, volume > 1.5x average, and price > 1d EMA(50)
# Short when price breaks below Donchian low (20) on 4h, volume > 1.5x average, and price < 1d EMA(50)
# Exit when price crosses back through Donchian midpoint or EMA(50)
# Uses volume and EMA to filter false breakouts, targeting 75-200 trades over 4 years
# Works in bull markets (breakouts) and bear markets (breakdowns) with volume confirmation

name = "4h_donchian20_12h_vol_1d_ema_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12h volume confirmation (volume > 1.5x 20-period average)
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_threshold = 1.5 * vol_ma_12h_aligned
    
    # 1d EMA(50) filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_threshold[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses back through midpoint or EMA
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or close[i] <= ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or close[i] >= ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts/breakdowns with volume confirmation and EMA filter
            # Bullish breakout: price > Donchian high + volume > threshold + price > EMA
            if (close[i] > donchian_high[i] and 
                volume[i] > volume_threshold[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakdown: price < Donchian low + volume > threshold + price < EMA
            elif (close[i] < donchian_low[i] and 
                  volume[i] > volume_threshold[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals