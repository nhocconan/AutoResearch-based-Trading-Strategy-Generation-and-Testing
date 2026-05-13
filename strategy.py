#!/usr/bin/env python3
# 12h_Donchian_Breakout_Volume_Trend
# Hypothesis: Breakout of 20-period Donchian channel with volume confirmation and 1d trend filter.
# Works in bull markets by capturing breakouts, in bear markets by avoiding false breakouts via trend filter.
# Volume confirmation reduces false signals. Trend filter ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (<30/year) to minimize fee drag on 12h timeframe.

name = "12h_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
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

    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    
    # Donchian channels (20-period) on 12h high/low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: 20-period rolling max of high
    upperband = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period rolling min of low
    lowerband = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Align indicators to 12h timeframe
    upperband_aligned = align_htf_to_ltf(prices, df_12h, upperband)
    lowerband_aligned = align_htf_to_ltf(prices, df_12h, lowerband)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required value is NaN
        if (np.isnan(upperband_aligned[i]) or np.isnan(lowerband_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Breakout conditions
        bullish_breakout = high[i] > upperband_aligned[i]
        bearish_breakout = low[i] < lowerband_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]

        if position == 0:
            # LONG: Bullish breakout + volume confirmation + uptrend
            if bullish_breakout and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish breakout + volume confirmation + downtrend
            elif bearish_breakout and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel (below upper band) OR trend reversal
            if close[i] < upperband_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel (above lower band) OR trend reversal
            if close[i] > lowerband_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals