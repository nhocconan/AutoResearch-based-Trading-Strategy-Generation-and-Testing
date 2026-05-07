#!/usr/bin/env python3

name = "1d_Camarilla_R1S1_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous weekly bar
    high_1w_prev = df_1w['high'].values[:-1]
    low_1w_prev = df_1w['low'].values[:-1]
    close_1w_prev = df_1w['close'].values[:-1]
    high_1w_prev = np.concatenate([[np.nan], high_1w_prev])
    low_1w_prev = np.concatenate([[np.nan], low_1w_prev])
    close_1w_prev = np.concatenate([[np.nan], close_1w_prev])
    
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    high_low = high_1w_prev - low_1w_prev
    r1 = close_1w_prev + 1.1 * high_low / 12
    s1 = close_1w_prev - 1.1 * high_low / 12
    
    # Align weekly indicators to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume filter: current volume > 1.5x 20-period average (daily)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 20  # ~20 days to reduce trades
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        trend_1w_up = close_1w_aligned[i] > ema_20_1w_aligned[i]
        trend_1w_down = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: break above R1 with volume in weekly uptrend
            if (close[i] > r1_aligned[i] and 
                trend_1w_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: break below S1 with volume in weekly downtrend
            elif (close[i] < s1_aligned[i] and 
                  trend_1w_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: close back below S1 or trend change
            if (close[i] < s1_aligned[i]) or not trend_1w_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close back above R1 or trend change
            if (close[i] > r1_aligned[i]) or not trend_1w_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 1w EMA20 trend filter and volume confirmation on daily timeframe.
# Long when price breaks above R1 with volume spike in weekly uptrend.
# Short when price breaks below S1 with volume spike in weekly downtrend.
# Exits when price returns to S1/R1 or trend changes.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume confirmation avoids false breakouts. Cooldown (20 days) reduces trade frequency.
# Target: 10-20 trades/year. Works in bull markets by catching breakouts in uptrends
# and in bear markets by shorting breakdowns in downtrends. Daily timeframe with weekly filter
# provides balance between signal quality and trade frequency. Camarilla levels provide precise
# support/resistance derived from prior week's range.