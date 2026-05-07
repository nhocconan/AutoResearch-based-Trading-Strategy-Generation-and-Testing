# -*- coding: utf-8 -*-
#!/usr/bin/env python3

name = "6h_Keltner_MeanReversion_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Keltner channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(10) for Keltner channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(10) using Wilder's smoothing
    atr_10 = np.full(len(tr), np.nan)
    for i in range(len(tr)):
        if i < 10:
            atr_10[i] = np.nan
        elif i == 10:
            atr_10[i] = np.nanmean(tr[1:11])
        else:
            atr_10[i] = (atr_10[i-1] * 9 + tr[i]) / 10
    
    # Keltner Channel: EMA(20) ± 2 * ATR(10)
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_20 + 2.0 * atr_10
    keltner_lower = ema_20 - 2.0 * atr_10
    
    # Align Keltner channels to 6h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (on 6h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # Prevent overtrading (approx 1 day for 6h)
    
    start_idx = max(20, 50)  # Warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(keltner_upper_aligned[i]) or 
            np.isnan(keltner_lower_aligned[i]) or 
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
            # Mean reversion long: price touches lower Keltner band in daily uptrend with volume
            if (low[i] <= keltner_lower_aligned[i] and 
                trend_1d_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Mean reversion short: price touches upper Keltner band in daily downtrend with volume
            elif (high[i] >= keltner_upper_aligned[i] and 
                  trend_1d_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price returns to EMA(20) or trend changes
            if (close[i] >= ema_20_aligned[i] or not trend_1d_up):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to EMA(20) or trend changes
            if (close[i] <= ema_20_aligned[i] or not trend_1d_down):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Align EMA20 for exit condition
ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20) if 'df_1d' in locals() else np.full(n, np.nan)