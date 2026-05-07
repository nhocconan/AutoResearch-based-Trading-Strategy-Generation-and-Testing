#!/usr/bin/env python3
name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Weekly OHLC for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = weekly_close + (weekly_high - weekly_low) * 1.1 / 12
    camarilla_s1 = weekly_close - (weekly_high - weekly_low) * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Weekly trend: price above/below EMA34
    ema_34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    weekly_trend_up = close > ema_34_aligned
    weekly_trend_down = close < ema_34_aligned
    
    # Volume filter: current volume > 2.0x 15-period average (approx 3 weeks)
    vol_ma_15 = np.full(n, np.nan)
    for i in range(15, n):
        vol_ma_15[i] = np.mean(volume[i-15:i])
    vol_filter = volume > (2.0 * vol_ma_15)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 30  # ~1 month (30*1d) to prevent overtrading
    
    start_idx = 15  # Volume MA needs 15 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_15[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        trend_up = weekly_trend_up[i]
        trend_down = weekly_trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Camarilla R1 with volume in weekly uptrend
            if (close[i] > r1_aligned[i] and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S1 with volume in weekly downtrend
            elif (close[i] < s1_aligned[i] and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Camarilla S1 or weekly trend changes to down
            if close[i] < s1_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above Camarilla R1 or weekly trend changes to up
            if close[i] > r1_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 1d timeframe, price breaking above/below weekly Camarilla R1/S1 levels with volume confirmation and weekly EMA34 trend filter captures institutional breakout momentum. Weekly Camarilla levels provide mathematically derived support/resistance with institutional relevance. Works in bull markets (breakouts above R1 in weekly uptrend) and bear markets (breakdowns below S1 in weekly downtrend). Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag while capturing significant moves. Weekly trend filter ensures alignment with higher timeframe momentum.