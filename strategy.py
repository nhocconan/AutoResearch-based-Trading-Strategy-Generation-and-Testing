#!/usr/bin/env python3
# 12h_Donchian20_1dTrend_VolumeFilter
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long: price breaks above Donchian upper band + 1d close > EMA50 + volume > 1.5x 20-period average.
# Short: price breaks below Donchian lower band + 1d close < EMA50 + volume > 1.5x 20-period average.
# Exit: trend reversal (price crosses EMA50 on 1d) or opposite Donchian break.
# Works in bull/bear by following 1d trend direction. Target: 15-30 trades/year per symbol.

name = "12h_Donchian20_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume filter: 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    volume_threshold = vol_ma * 1.5
    
    # Align 1d EMA50 trend to 12h
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema50_aligned[i]
        bearish_trend = close[i] < ema50_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > volume_threshold[i]
        
        if position == 0:
            # Enter long: bullish trend + price breaks above Donchian upper + volume
            if bullish_trend and close[i] > highest_high[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price breaks below Donchian lower + volume
            elif bearish_trend and close[i] < lowest_low[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend reversal or price breaks below Donchian lower
            if bearish_trend or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend reversal or price breaks above Donchian upper
            if bullish_trend or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals