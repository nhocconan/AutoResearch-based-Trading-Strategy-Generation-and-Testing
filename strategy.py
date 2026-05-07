#!/usr/bin/env python3

name = "12h_KAMA_Direction_1dTrend_VolumeFilter"
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
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # KAMA calculation on 12h close (trend direction)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # temporary, will fix in loop
    # Recalculate volatility properly
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.zeros(n)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if close > KAMA, -1 if close < KAMA
    kama_direction = np.where(close > kama, 1, -1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Daily trend: 1 for up, -1 for down
    daily_trend = np.where(close_1d_aligned > ema_34_1d_aligned, 1, -1)
    
    # Volume filter: current volume > 2.0x 20-period average (on 12h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # Prevent overtrading (approx 2 days for 12h)
    
    start_idx = max(20, 34)  # Warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_direction[i]) or 
            np.isnan(daily_trend[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: KAMA up AND daily trend up AND volume filter
            if (kama_direction[i] == 1 and 
                daily_trend[i] == 1 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: KAMA down AND daily trend down AND volume filter
            elif (kama_direction[i] == -1 and 
                  daily_trend[i] == -1 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: KAMA turns down OR daily trend changes
            if (kama_direction[i] == -1) or (daily_trend[i] == -1):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA turns up OR daily trend changes
            if (kama_direction[i] == 1) or (daily_trend[i] == 1):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
# in ranging markets it stays flat. We use KAMA direction as trend filter on 12h timeframe.
# Entry only when both 12h KAMA trend and daily trend align (same direction) with volume confirmation.
# This avoids whipsaws in ranging markets while capturing strong trends.
# Works in bull markets (captures uptrends) and bear markets (captures downtrends).
# Volume filter ensures institutional participation. Cooldown prevents overtrading.