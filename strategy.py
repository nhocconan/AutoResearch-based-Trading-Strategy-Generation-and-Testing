#!/usr/bin/env python3
# 4h_DonchianBreakout_1dTrend_Volume
# Hypothesis: Use Donchian channel breakout on 4h with 1d trend filter (EMA50) and volume confirmation.
# Long when price breaks above upper Donchian(20) + price > 1d EMA50 + volume spike.
# Short when price breaks below lower Donchian(20) + price < 1d EMA50 + volume spike.
# Exit when price returns to Donchian midpoint or trend fails.
# Designed for low frequency (20-50 trades/year) to avoid fee drag. Works in bull (catch breakouts)
# and bear (catch breakdowns) with trend filter and volume confirmation.

name = "4h_DonchianBreakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
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
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i]) or np.isnan(vol_ma_20[i])):
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
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        return_to_mid = abs(close[i] - donchian_mid[i]) < 0.1 * (highest_high[i] - lowest_low[i])
        
        if position == 0:
            # LONG: breakout above upper Donchian + uptrend + volume
            if breakout_up and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: breakout below lower Donchian + downtrend + volume
            elif breakout_down and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: return to midpoint or trend fails
            if return_to_mid or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: return to midpoint or trend fails
            if return_to_mid or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals