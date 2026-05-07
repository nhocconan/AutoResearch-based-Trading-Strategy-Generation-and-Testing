#!/usr/bin/env python3

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_R1 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_S1 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume filter: current volume > 1.5x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~1 day for 4h
    
    start_idx = max(40, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 12h trend direction
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['close'].values)
        trend_12h_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_12h_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above R1 with volume in 12h uptrend
            if (close[i] > R1_aligned[i] and 
                close[i-1] <= R1_aligned[i-1] and 
                trend_12h_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S1 with volume in 12h downtrend
            elif (close[i] < S1_aligned[i] and 
                  close[i-1] >= S1_aligned[i-1] and 
                  trend_12h_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Close below S1 OR trend change
            if (close[i] < S1_aligned[i]) or not trend_12h_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R1 OR trend change
            if (close[i] > R1_aligned[i]) or not trend_12h_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 12h trend filter and volume confirmation on 4h timeframe.
# Long when price breaks above R1 level with volume spike in 12h uptrend.
# Short when price breaks below S1 level with volume spike in 12h downtrend.
# Exits on reversal to S1/R1 levels or trend change.
# Uses 1d Camarilla levels for intraday support/resistance and 12h EMA50 for trend.
# Volume confirmation filters false breakouts. Cooldown prevents overtrading.
# Target: 20-40 trades/year to avoid fee drag. Works in bull/bear by capturing
# significant intraday moves with trend alignment. 4h timeframe reduces noise vs 12h.