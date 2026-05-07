# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Breakouts beyond Camarilla R3/S3 levels on 4h timeframe, confirmed by daily trend and volume spikes, capture strong momentum moves in both bull and bear markets. The strategy targets fewer than 400 total trades over 4 years to minimize fee drag, using a daily EMA trend filter and volume confirmation to ensure trades align with higher timeframe momentum and avoid false breakouts. Works in bull markets by capturing upside breakouts and in bear markets by capturing downside breakdowns.

#!/usr/bin/env python3

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_val = high_prev - low_prev
    
    R3 = pivot + (high_prev - low_prev) * 1.1 / 4
    S3 = pivot - (high_prev - low_prev) * 1.1 / 4
    
    # Align daily levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 2x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~1 day for 4h
    
    start_idx = max(30, 34)  # Warmup for daily data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above R3 level in daily uptrend with volume spike
            if (close[i] > R3_aligned[i] and 
                trend_1d_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below S3 level in daily downtrend with volume spike
            elif (close[i] < S3_aligned[i] and 
                  trend_1d_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price closes below pivot OR trend change
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
            if (close[i] < pivot_aligned[i]) or not trend_1d_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above pivot OR trend change
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
            if (close[i] > pivot_aligned[i]) or not trend_1d_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout on 4h timeframe, confirmed by daily trend and volume spike.
# Long when price breaks above R3 level in daily uptrend with volume spike (>2x average).
# Short when price breaks below S3 level in daily downtrend with volume spike.
# Daily EMA34 filter ensures we trade with the higher timeframe trend.
# Volume confirmation filters out false breakouts. Cooldown prevents overtrading.
# Target: 20-40 trades/year to avoid fee drag. Works in both bull and bear markets by capturing breakouts in direction of daily trend.