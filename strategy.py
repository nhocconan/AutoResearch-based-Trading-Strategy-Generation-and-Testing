#!/usr/bin/env python3

name = "4h_Trix_Slope_Change_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate TRIX on daily close (12-period EMA smoothed 3 times)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False).mean().values
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # TRIX slope (1-bar change)
    trix_slope = np.zeros_like(trix)
    trix_slope[1:] = trix[1:] - trix[:-1]
    
    # Align TRIX and slope to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_slope_aligned = align_htf_to_ltf(prices, df_1d, trix_slope)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.8x 20-period average (on 4h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # ~1.3 days for 4h
    
    start_idx = max(20, 34, 40)  # Warmup for volume, EMA, and TRIX
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(trix_slope_aligned[i]) or 
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
        trend_up = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
        trend_down = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: TRIX slope turns positive in uptrend with volume
            if (trix_slope_aligned[i] > 0 and 
                trix_slope_aligned[i-1] <= 0 and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: TRIX slope turns negative in downtrend with volume
            elif (trix_slope_aligned[i] < 0 and 
                  trix_slope_aligned[i-1] >= 0 and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: TRIX slope turns negative OR trend change
            if (trix_slope_aligned[i] < 0) or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX slope turns positive OR trend change
            if (trix_slope_aligned[i] > 0) or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX slope changes capture momentum shifts in higher timeframe trends.
# Long when TRIX slope turns positive in daily uptrend with volume confirmation.
# Short when TRIX slope turns negative in daily downtrend with volume confirmation.
# Daily EMA34 ensures we trade with the higher timeframe trend direction.
# Volume filter confirms institutional participation. Cooldown prevents overtrading.
# Using 4h timeframe balances responsiveness and trade frequency. Target: 25-50 trades/year.