#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Williams %R (14) with 1-week EMA(50) trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions: <-80 oversold (long), >-20 overbought (short).
# The 1-week EMA(50) filters trades to follow the dominant weekly trend, reducing whipsaws in ranging markets.
# Volume > 1.5x the 20-day average confirms institutional participation and reduces false signals.
# This strategy targets 15-25 trades per year per symbol (60-100 total over 4 years), staying within optimal range to minimize fee drag.
# Works in both bull and bear markets: the trend filter ensures alignment with higher-timeframe momentum,
# while Williams %R captures mean-reversion entries during pullbacks or bounces.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1-week EMA(50) for trend filter
    ema_len = 50
    if len(df_1w) < ema_len:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams %R (14) on daily
    wr_len = 14
    if len(high) < wr_len:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=wr_len, min_periods=wr_len).max().values
    lowest_low = pd.Series(low).rolling(window=wr_len, min_periods=wr_len).min().values
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    williams_r = -100 * (highest_high - close) / range_hl
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, wr_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1-week EMA50
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + above weekly EMA + volume
            if (williams_r[i] < -80 and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R overbought (> -20) + below weekly EMA + volume
            elif (williams_r[i] > -20 and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to -50 or breaks below weekly EMA
            if williams_r[i] > -50 or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to -50 or breaks above weekly EMA
            if williams_r[i] < -50 or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_WilliamsR_EMA50_Volume_v1"
timeframe = "1d"
leverage = 1.0