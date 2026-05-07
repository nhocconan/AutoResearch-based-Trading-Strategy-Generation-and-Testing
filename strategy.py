#!/usr/bin/env python3

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using previous day's HLC)
    # We need previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (on 4h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # Prevent overtrading (approx 1 day for 4h)
    
    start_idx = max(20, 34)  # Warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
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
            # Long: price breaks above Camarilla R1 in daily uptrend with volume confirmation
            if (close[i] > R1_aligned[i] and 
                trend_1d_up and 
                vol_filter[i]):
                signals[i] = 0.30
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below Camarilla S1 in daily downtrend with volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  trend_1d_down and 
                  vol_filter[i]):
                signals[i] = -0.30
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price closes below Camarilla R1 OR trend change
            if (close[i] < R1_aligned[i]) or not trend_1d_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price closes above Camarilla S1 OR trend change
            if (close[i] > S1_aligned[i]) or not trend_1d_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Breakout of Camarilla R1/S1 levels on 4h timeframe, confirmed by daily trend and volume spike.
# Long when price breaks above R1 in daily uptrend with volume spike.
# Short when price breaks below S1 in daily downtrend with volume spike.
# Daily EMA34 filter ensures we trade with the higher timeframe trend.
# Volume confirmation filters out false breakouts. Cooldown prevents overtrading.
# Target: 20-50 trades/year to avoid fee drag. Works in both bull and bear markets by capturing breakouts in direction of daily trend.