#!/usr/bin/env python3
# 4h_DonchianBreakout_1dTrend_Volume
# Hypothesis: Use Donchian channel breakout (20-period) on 4h for entry, filtered by 1d EMA50 trend and volume spike.
# Long when price breaks above upper Donchian band and close > 1d EMA50, short when breaks below lower band and close < 1d EMA50.
# Exit on opposite Donchian breakout or trend failure. Designed for low frequency (20-50 trades/year) to avoid fee drag.
# Works in bull (catch breakouts) and bear (catch breakdowns) with trend filter and volume confirmation.

name = "4h_DonchianBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period=20):
    """
    Calculate Donchian channels.
    Returns upper band, lower band, and middle band.
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels on 4h data (20-period)
    upper, lower, middle = donchian_channels(high, low, 20)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Donchian breakout signals
        buy_signal = close[i] > upper[i]  # Break above upper band
        sell_signal = close[i] < lower[i]  # Break below lower band
        
        if position == 0:
            # LONG: Price breaks above upper Donchian band, trend up, volume confirmation
            if buy_signal and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian band, trend down, volume confirmation
            elif sell_signal and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian band or trend fails
            if sell_signal or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian band or trend fails
            if buy_signal or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals