#!/usr/bin/env python3
# 12h_Donchian_Breakout_1dTrend_Volume
# Hypothesis: Use 12h Donchian channel breakout with 1d EMA50 trend filter and volume spike.
# Long when price breaks above Donchian upper band and price > 1d EMA50 with volume > 1.5x average.
# Short when price breaks below Donchian lower band and price < 1d EMA50 with volume > 1.5x average.
# Exit when price crosses back through Donchian middle (20-period average of high/low).
# Designed for low frequency (15-35 trades/year) to avoid fee drag. Works in bull (catch breakouts)
# and bear (catch breakdowns) with trend filter and volume confirmation.

name = "12h_Donchian_Breakout_1dTrend_Volume"
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
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 12h data
    period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= period - 1:
            highest_high[i] = np.max(high[i - period + 1:i + 1])
            lowest_low[i] = np.min(low[i - period + 1:i + 1])
    
    # Middle band (average of upper and lower)
    middle_band = (highest_high + lowest_low) / 2
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    # Align daily data to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period)  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
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
        vol_ok = volume[i] > vol_threshold[i]
        
        # Donchian breakout signals
        buy_breakout = close[i] > highest_high[i]
        sell_breakout = close[i] < lowest_low[i]
        
        # Exit signals: price crosses back through middle band
        exit_long = close[i] < middle_band[i]
        exit_short = close[i] > middle_band[i]
        
        if position == 0:
            # LONG: price breaks above Donchian upper band, price > daily EMA50, volume confirmation
            if buy_breakout and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band, price < daily EMA50, volume confirmation
            elif sell_breakout and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price crosses below Donchian middle band
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above Donchian middle band
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals