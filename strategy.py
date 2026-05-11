#!/usr/bin/env python3
# 4h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: 4-hour Donchian(20) breakout with 1-day trend filter and volume confirmation. Works in bull via breakout above upper band in uptrend, and in bear via breakout below lower band in downtrend. Uses volume surge to avoid false breakouts. Designed for 20-50 trades/year to minimize fee drag.

name = "4h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4-hour OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily trend: 50-period EMA slope
    daily_close = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope_50_1d = np.diff(ema_50_1d, prepend=ema_50_1d[0])
    ema_slope_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_50_1d)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop-loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(ema_slope_50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from daily EMA50 slope
        bullish_trend = ema_slope_50_1d_aligned[i] > 0
        bearish_trend = ema_slope_50_1d_aligned[i] < 0
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above upper Donchian in bullish trend with volume surge
            if close[i] > highest_high[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower Donchian in bearish trend with volume surge
            elif close[i] < lowest_low[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit: close below lower Donchian
                if close[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: close above upper Donchian
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals