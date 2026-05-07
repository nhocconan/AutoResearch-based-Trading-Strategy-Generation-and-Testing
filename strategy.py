#!/usr/bin/env python3

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
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
    
    # Get 4h data for Camarilla levels and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1_4h = close_4h + 1.1 * range_4h / 12
    s1_4h = close_4h - 1.1 * range_4h / 12
    
    # Align Camarilla levels to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume filter: current volume > 1.5x 20-period average (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~3 hours cooldown
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 4h trend direction
        trend_4h_up = close[i] > ema_50_4h_aligned[i]
        trend_4h_down = close[i] < ema_50_4h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            if session_filter[i] and vol_filter[i]:
                # Long: Break above R1 in 4h uptrend
                if close[i] > r1_4h_aligned[i] and trend_4h_up:
                    signals[i] = 0.20
                    position = 1
                    bars_since_last_trade = 0
                # Short: Break below S1 in 4h downtrend
                elif close[i] < s1_4h_aligned[i] and trend_4h_down:
                    signals[i] = -0.20
                    position = -1
                    bars_since_last_trade = 0
        elif position == 1:
            # Exit: Break below S1 or trend change
            if close[i] < s1_4h_aligned[i] or not trend_4h_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Break above R1 or trend change
            if close[i] > r1_4h_aligned[i] or not trend_4h_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and volume confirmation captures institutional breakouts in both bull and bear markets. The 4h trend filter ensures we trade with the higher-timeframe momentum, while volume confirms participation. Session filter (08-20 UTC) reduces noise during low-liquidity hours. Target: 15-35 trades/year. Works in bull markets by buying R1 breaks in uptrends and in bear markets by selling S1 breaks in downtrends. Camarilla levels provide precise intraday support/resistance, EMA50 establishes trend, and volume avoids false breakouts. This avoids overtrading by requiring multiple confirmations and cooldown periods.