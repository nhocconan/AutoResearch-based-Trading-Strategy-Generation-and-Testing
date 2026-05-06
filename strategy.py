#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot breakouts with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above weekly Camarilla R4 AND 1d EMA34 is rising AND 6h volume > 1.5 * avg_volume(20)
# Short when price breaks below weekly Camarilla S4 AND 1d EMA34 is falling AND 6h volume > 1.5 * avg_volume(20)
# Exit when price returns to weekly Camarilla midpoint (PP)
# Uses discrete sizing 0.25 to control drawdown and fee churn
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Camarilla provides strong institutional levels from weekly structure
# 1d EMA34 ensures we trade with the daily trend while reducing whipsaws in 6h timeframe
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "6h_1wCamarilla_R4_S4_Breakout_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least 1 completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Range = High - Low
    range_1w = high_1w - low_1w
    # Camarilla levels:
    # R4 = PP + (Range * 1.1/2)
    # S4 = PP - (Range * 1.1/2)
    r4_1w = pp_1w + (range_1w * 1.1 / 2.0)
    s4_1w = pp_1w - (range_1w * 1.1 / 2.0)
    pp_1w_level = pp_1w  # Midpoint for exit
    
    # Align 1w Camarilla levels to 6h timeframe (wait for completed 1w bar)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w_level)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:  # Need at least 30 completed 6h bars for volume avg
        return np.zeros(n)
    volume_6h = df_6h['volume'].values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_6h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R4, EMA34 rising, volume spike
            if (close[i] > r4_1w_aligned[i] and close[i-1] <= r4_1w_aligned[i-1] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S4, EMA34 falling, volume spike
            elif (close[i] < s4_1w_aligned[i] and close[i-1] >= s4_1w_aligned[i-1] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly Camarilla midpoint (PP)
            if close[i] <= pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly Camarilla midpoint (PP)
            if close[i] >= pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals