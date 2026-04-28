#!/usr/bin/env python3
"""
6h_MACD_Trend_Filter_12hVolumeSpike
Hypothesis: MACD histogram cross with 12h EMA50 trend filter and 12h volume spike (2x) captures strong momentum moves while avoiding whipsaws. Works in bull (uptrend + volume) and bear (downtrend + volume) by filtering with 12h EMA50. Targets 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and volume filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h volume 20-period MA for spike detection
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # MACD on 6h close (fast=12, slow=26, signal=9)
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close_series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(macd_hist[i]) or
            np.isnan(macd_hist[i-1])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: 12h volume > 2x 20-period MA
        vol_spike = volume_12h_aligned[i] > (2.0 * vol_ma_20_12h_aligned[i])
        
        # MACD histogram cross: bullish when crosses above zero, bearish when below
        macd_bullish_cross = (macd_hist[i-1] <= 0) and (macd_hist[i] > 0)
        macd_bearish_cross = (macd_hist[i-1] >= 0) and (macd_hist[i] < 0)
        
        # Entry logic: MACD cross in direction of trend with volume spike
        long_entry = vol_spike and uptrend and macd_bullish_cross
        short_entry = vol_spike and downtrend and macd_bearish_cross
        
        # Exit logic: opposite MACD cross or trend change
        long_exit = macd_bearish_cross or (not uptrend)
        short_exit = macd_bullish_cross or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_MACD_Trend_Filter_12hVolumeSpike"
timeframe = "6h"
leverage = 1.0