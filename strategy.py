#!/usr/bin/env python3
# 4h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: 4h breakout at Donchian(20) channel with daily trend filter and volume confirmation.
# Uses 1d EMA34 trend for direction, Donchian(20) from 4h for entry levels, and volume spike for confirmation.
# Designed for low trade frequency (20-50/year) to minimize fee drift while capturing strong moves.

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
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Donchian(20) levels from 4h ---
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # --- Daily trend: EMA34 slope ---
    daily_close = df_1d['close'].values
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slope_34_1d = np.diff(ema_34_1d, prepend=ema_34_1d[0])
    ema_slope_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_34_1d)
    
    # --- ATR for volatility and trailing stop ---
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # --- Volume confirmation (2.5x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup: ensure we have enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(ema_slope_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from daily EMA34 slope
        bullish_trend = ema_slope_34_1d_aligned[i] > 0
        bearish_trend = ema_slope_34_1d_aligned[i] < 0
        
        # Volume confirmation (2.5x average)
        volume_surge = volume[i] > 2.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian in bullish trend with volume surge
            if close[i] > upper[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below lower Donchian in bearish trend with volume surge
            elif close[i] < lower[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Trailing stop: exit if price drops 3.0*ATR from highest high
                if close[i] < highest_high_since_entry - 3.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Trailing stop: exit if price rises 3.0*ATR from lowest low
                if close[i] > lowest_low_since_entry + 3.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals