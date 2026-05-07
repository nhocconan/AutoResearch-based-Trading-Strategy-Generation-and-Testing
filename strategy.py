#!/usr/bin/env python3

name = "12h_TRIX_VolumeSpike_TrendFilter"
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
    
    # Get daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 18:
        return np.zeros(n)
    
    # Calculate TRIX (15-period triple EMA) on daily closes
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = np.where(ema3[:-1] != 0, (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100, 0)
    trix = np.concatenate([np.zeros(1), trix_raw])  # align with original length
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 2.5x 20-period average (on 12h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # Prevent overtrading (approx 2 days for 12h)
    
    start_idx = max(20, 34)  # Warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_aligned[i]) or 
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
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: TRIX crosses above zero in daily uptrend with volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                trend_1d_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: TRIX crosses below zero in daily downtrend with volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  trend_1d_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: TRIX crosses below zero OR trend change
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or not trend_1d_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero OR trend change
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or not trend_1d_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX (15-period triple EMA) on daily timeframe filters momentum with reduced lag.
# Long when TRIX crosses above zero in daily uptrend, short when crosses below zero in daily downtrend.
# Volume confirmation ensures institutional participation. Cooldown prevents overtrading.
# Effective in both bull (captures momentum) and bear (avoids false signals via trend filter).