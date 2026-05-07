#!/usr/bin/env python3
name = "1h_4h1d_Trend_Volume_Confluence_v1"
timeframe = "1h"
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
    
    # Get 4h and 1d data for trend and volume filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA20 trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d EMA20 trend filter
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 1d volume filter: current 1d volume > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d, additional_delay_bars=0)
    vol_filter_1d = df_1d['volume'].values > (1.5 * vol_ma_20_1d)
    vol_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d.astype(float), additional_delay_bars=0)
    
    # 1h volume filter for entry timing
    vol_ma_20_1h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_1h[i] = np.mean(volume[i-20:i])
    vol_filter_1h = volume > (1.5 * vol_ma_20_1h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # 6 hours cooldown
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(vol_filter_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction from 4h and 1d
        trend_up = (close > ema_20_4h_aligned[i]) and (close > ema_20_1d_aligned[i])
        trend_down = (close < ema_20_4h_aligned[i]) and (close < ema_20_1d_aligned[i])
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above recent high with volume in uptrend
            if (close[i] > np.max(high[i-20:i]) and 
                trend_up and 
                vol_filter_1d_aligned[i] and 
                vol_filter_1h[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below recent low with volume in downtrend
            elif (close[i] < np.min(low[i-20:i]) and 
                  trend_down and 
                  vol_filter_1d_aligned[i] and 
                  vol_filter_1h[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below recent low or trend changes
            if close[i] < np.min(low[i-20:i]) or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price rises above recent high or trend changes
            if close[i] > np.max(high[i-20:i]) or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h trend following with 4h/1d EMA20 trend filter and volume confirmation.
# Long when price breaks above 20-period high in uptrend (4h & 1d) with volume confirmation.
# Short when price breaks below 20-period low in downtrend (4h & 1d) with volume confirmation.
# Uses 1h for entry timing, 4h/1d for signal direction to reduce trade frequency.
# Session filter (08-20 UTC) to avoid low-volume periods.
# Position size fixed at 0.20 to manage risk.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Target: 60-150 total trades over 4 years (15-37/year) as per experiment guidelines.