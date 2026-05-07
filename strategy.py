#!/usr/bin/env python3

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla R1 and S1 from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_R1 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_S1 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume filter: current volume > 1.5x 20-period average (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    # Breakout threshold: 0.5% above/below level
    breakout_threshold = 0.005
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # ~4 days for 12h to reduce trades
    
    start_idx = max(40, 34, 20)
    
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
        
        # Determine 1d trend direction
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above R1 with volume in 1d uptrend
            if (close[i] > R1_aligned[i] * (1 + breakout_threshold) and 
                close[i-1] <= R1_aligned[i-1] and 
                trend_1d_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S1 with volume in 1d downtrend
            elif (close[i] < S1_aligned[i] * (1 - breakout_threshold) and 
                  close[i-1] >= S1_aligned[i-1] and 
                  trend_1d_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Close below S1 OR trend change
            if (close[i] < S1_aligned[i]) or not trend_1d_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R1 OR trend change
            if (close[i] > R1_aligned[i]) or not trend_1d_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 1d trend filter and volume confirmation on 12h timeframe.
# Long when price breaks above R1 level (with 0.5% buffer) with volume spike in 1d uptrend.
# Short when price breaks below S1 level (with 0.5% buffer) with volume spike in 1d downtrend.
# Exits on reversal to S1/R1 levels or trend change.
# Uses 1d Camarilla levels for support/resistance and 1d EMA34 for trend.
# Volume confirmation filters false breakouts. Cooldown (4 days) and breakout threshold
# reduce trade frequency to avoid fee drag. Target: 15-30 trades/year to work in bull/bear markets
# by capturing significant daily moves with trend alignment. 12h timeframe reduces noise vs 1d.