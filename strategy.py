#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-period high AND close > 1d EMA50 AND volume > 1.5x average
# Short when price breaks below 20-period low AND close < 1d EMA50 AND volume > 1.5x average
# Exit when price crosses 10-period EMA (trend reversal) OR opposite Donchian breakout
# Uses 12h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with daily trend filter for BTC/ETH resilience.
# Donchian provides price structure; EMA50 filters trend; volume confirms breakout authenticity.

name = "12h_Donchian20_1dEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels on 12h data (using previous bar's OHLC to avoid look-ahead)
    if len(high_12h) >= 20:
        # Use rolling window on 12h data
        high_ma_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        low_ma_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    else:
        high_ma_20 = np.full_like(high_12h, np.nan)
        low_ma_20 = np.full_like(low_12h, np.nan)
    
    # Align Donchian levels to 12h timeframe (already aligned since calculated on 12h)
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_ma_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_ma_20)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 12h volume > 1.5x 20-period average (spike confirmation)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_12h)
    
    # Exit condition: 10-period EMA on 12h close for trend reversal
    ema10_12h = pd.Series(close_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_12h_aligned = align_htf_to_ltf(prices, df_12h, ema10_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA and Donchian
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema10_12h_aligned[i]) or 
            np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > 20-period high AND close > 1d EMA50 AND volume spike
            if close[i] > high_20_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < 20-period low AND close < 1d EMA50 AND volume spike
            elif close[i] < low_20_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < 10-period EMA (trend reversal) OR price < 20-period low (opposite breakout)
            if close[i] < ema10_12h_aligned[i] or close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > 10-period EMA (trend reversal) OR price > 20-period high (opposite breakout)
            if close[i] > ema10_12h_aligned[i] or close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals