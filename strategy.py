#!/usr/bin/env python3

name = "4h_NewHighLowBreakout_1dTrend_Volume"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # 4-period high/low for breakout levels (lookback on completed bars)
    high_4 = np.full(n, np.nan)
    low_4 = np.full(n, np.nan)
    for i in range(4, n):
        high_4[i] = np.max(high[i-4:i])
        low_4[i] = np.min(low[i-4:i])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # ~1.3 days for 4h
    
    start_idx = max(25, 30)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_4[i]) or 
            np.isnan(low_4[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
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
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above 4-period high with volume in daily uptrend
            if (close[i] > high_4[i] and 
                close[i-1] <= high_4[i-1] and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below 4-period low with volume in daily downtrend
            elif (close[i] < low_4[i] and 
                  close[i-1] >= low_4[i-1] and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Close below 4-period low OR trend change
            if (close[i] < low_4[i]) or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above 4-period high OR trend change
            if (close[i] > high_4[i]) or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Breakout of short-term (4-period) high/low with daily trend filter and volume confirmation.
# Long when price breaks above 4-period high with volume spike in daily uptrend.
# Short when price breaks below 4-period low with volume spike in daily downtrend.
# Exits on reversal to 4-period low/high or trend change.
# Uses daily EMA50 for trend alignment and volume confirmation to filter false breakouts.
# Cooldown prevents overtrading. Target: 20-40 trades/year to avoid fee drag.
# Works in bull/bear by capturing significant intraday moves with trend alignment.