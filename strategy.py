#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla H3/L3 breakout with volume confirmation and 4h trend filter
    # Uses 4h EMA50 for trend direction (long only when price > EMA50, short only when price < EMA50)
    # 1h Camarilla levels provide precise entry/exit levels for mean reversion in ranging markets
    # Volume > 1.5x 20-period MA confirms institutional participation
    # Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
    # Discrete sizing 0.20 to minimize fee churn. Target: 15-30 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h_50 = np.full(len(close_4h), np.nan)
    for i in range(49, len(close_4h)):
        if i == 49:
            ema_4h_50[i] = np.mean(close_4h[i-49:i+1])
        else:
            ema_4h_50[i] = (close_4h[i] * 2 / (50 + 1)) + ema_4h_50[i-1] * (1 - 2 / (50 + 1))
    
    # Align 4h EMA50 to 1h timeframe
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla H3 and L3 levels
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align to 1h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA50
        uptrend = close[i] > ema_4h_50_aligned[i]
        downtrend = close[i] < ema_4h_50_aligned[i]
        
        # Breakout conditions with volume confirmation
        breakout_up = close[i] > camarilla_h3_aligned[i]
        breakout_down = close[i] < camarilla_l3_aligned[i]
        
        # Entry conditions: breakout with volume confirmation and trend alignment
        long_entry = breakout_up and (vol_ratio[i] > 1.5) and uptrend
        short_entry = breakout_down and (vol_ratio[i] > 1.5) and downtrend
        
        # Exit conditions: price returns to midpoint between H3 and L3
        midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_breakout_vol_trend_v1"
timeframe = "1h"
leverage = 1.0