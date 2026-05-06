#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h ATR-based volatility breakout with 1d trend filter and volume confirmation
# Long when price breaks above 12h high + ATR(12h)*0.5 AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 1.3 * avg_volume(20) on 6h
# Short when price breaks below 12h low - ATR(12h)*0.5 AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 1.3 * avg_volume(20) on 6h
# Exit when price crosses the 12h midpoint (average of 12h high and low over last 20 periods)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# ATR-based breakout adapts to volatility regime, reducing false breakouts in low volatility
# 1d EMA34 trend filter ensures we trade with dominant daily trend
# Volume confirmation (1.3x) validates breakout strength while limiting overtrading

name = "6h_12hATRBreakout_1dEMA34_Trend_VolumeConfirm"
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
    
    # Get 12h data ONCE before loop for ATR calculation and breakout levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need at least 20 completed 12h bars for ATR20 and Donchian20
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(20)
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20_12h = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian(20) for breakout levels and midpoint
    highest_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    breakout_up_12h = highest_20_12h + (atr_20_12h * 0.5)
    breakout_dn_12h = lowest_20_12h - (atr_20_12h * 0.5)
    midpoint_12h = (highest_20_12h + lowest_20_12h) / 2.0
    
    # Align 12h indicators to 6h timeframe (wait for completed 12h bar)
    breakout_up_aligned = align_htf_to_ltf(prices, df_12h, breakout_up_12h)
    breakout_dn_aligned = align_htf_to_ltf(prices, df_12h, breakout_dn_12h)
    midpoint_aligned = align_htf_to_ltf(prices, df_12h, midpoint_12h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_dn_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h breakout level, 1d EMA34 > EMA34 previous (uptrend), volume confirmation, in session
            if (close[i] > breakout_up_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h breakout level, 1d EMA34 < EMA34 previous (downtrend), volume confirmation, in session
            elif (close[i] < breakout_dn_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 12h midpoint
            if close[i] < midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 12h midpoint
            if close[i] > midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals