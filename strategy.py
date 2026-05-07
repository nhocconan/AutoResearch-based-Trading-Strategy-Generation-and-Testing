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
    
    # 4h OHLC for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA50 trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    trend_up = close > ema_50_4h_aligned
    trend_down = close < ema_50_4h_aligned
    
    # Daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: current volume > 2.0x 12-period average (12h)
    vol_ma_12 = np.full(n, np.nan)
    for i in range(12, n):
        vol_ma_12[i] = np.mean(volume[i-12:i])
    vol_filter = volume > (2.0 * vol_ma_12)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~12 hours to prevent overtrading
    
    start_idx = 12  # Volume MA needs 12 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Camarilla R1 with volume in 4h uptrend and session
            if (close[i] > r1_aligned[i] and 
                trending_up and 
                vol_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S1 with volume in 4h downtrend and session
            elif (close[i] < s1_aligned[i] and 
                  trending_down and 
                  vol_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Camarilla S1 or 4h trend changes to down
            if close[i] < s1_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price rises back above Camarilla R1 or 4h trend changes to up
            if close[i] > r1_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: On 1h timeframe, price breaking above/below Camarilla R1/S1 levels with volume confirmation and 4h EMA50 trend filter captures institutional breakout momentum. Camarilla levels provide mathematically derived support/resistance with institutional relevance. Works in bull markets (breakouts above R1 in 4h uptrend) and bear markets (breakdowns below S1 in 4h downtrend). Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag while capturing significant moves. 4h trend filter ensures alignment with higher timeframe momentum.