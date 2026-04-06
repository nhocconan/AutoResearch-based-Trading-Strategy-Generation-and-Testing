#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA20 trend filter and volume confirmation.
# Long when price breaks above 20-period high AND above 1d EMA20 AND volume > average.
# Short when price breaks below 20-period low AND below 1d EMA20 AND volume > average.
# Uses 1d EMA for trend filter and 1d OHLC for Donchian calculation (high/low of past 20 days).
# Designed for low turnover (~20-40 trades/year) with clear trend-following logic.
# Works in bull markets via breakouts and in bear markets via short breakdowns.

name = "12h_donchian20_1d_ema20_vol_v1"
timeframe = "12h"
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
    
    # 1d data for EMA and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema_20 = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # 1d Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h (shifted by 1 day for prior day's values)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume moving average for filter (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low (trend reversal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high (trend reversal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if vol_filter:
                # Long breakout: price above Donchian high AND above 1d EMA20 (uptrend)
                if close[i] > donchian_high_aligned[i] and close[i] > ema_20_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: price below Donchian low AND below 1d EMA20 (downtrend)
                elif close[i] < donchian_low_aligned[i] and close[i] < ema_20_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals