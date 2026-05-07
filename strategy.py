#!/usr/bin/env python3

name = "4h_WeeklyDonchianBreakout_1dTrend_Volume"
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
    
    # Get weekly data for Donchian channels (10 weeks lookback)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly Donchian channels (10-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min with min_periods
    donchian_high = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    donchian_low = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    
    # Align Donchian levels to 4h timeframe
    dh_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~2 days for 4h
    
    start_idx = max(30, 50)  # Warmup for weekly/daily data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dh_aligned[i]) or 
            np.isnan(dl_aligned[i]) or 
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
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_1d_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above weekly Donchian high with volume in daily uptrend
            if (close[i] > dh_aligned[i] and 
                close[i-1] <= dh_aligned[i-1] and 
                trend_1d_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below weekly Donchian low with volume in daily downtrend
            elif (close[i] < dl_aligned[i] and 
                  close[i-1] >= dl_aligned[i-1] and 
                  trend_1d_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Close below weekly Donchian low OR trend change
            if (close[i] < dl_aligned[i]) or not trend_1d_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly Donchian high OR trend change
            if (close[i] > dh_aligned[i]) or not trend_1d_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout with daily trend filter and volume confirmation.
# Long when price breaks above 10-week high with volume spike in daily uptrend.
# Short when price breaks below 10-week low with volume spike in daily downtrend.
# Exits on reversal of weekly breakout level or trend change.
# Uses weekly structure for major trend and daily EMA50 for intermediate trend alignment.
# Volume confirmation filters false breakouts. Cooldown prevents overtrading.
# Target: 20-40 trades/year to avoid fee drift. Works in bull/bear by capturing
# major weekly breakouts with trend alignment. Weekly timeframe reduces noise.