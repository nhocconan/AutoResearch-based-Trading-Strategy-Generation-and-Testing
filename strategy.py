#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ATR-based volatility breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 1d high + 0.5*ATR(10) AND price > 12h EMA50 AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below 1d low - 0.5*ATR(10) AND price < 12h EMA50 AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses 12h EMA50 in opposite direction OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# ATR-based breakout captures momentum after volatility contraction/expansion cycles
# 12h EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume confirmation ensures breakout validity and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "12h_ATRBreakout_12hEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need enough for ATR calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(10)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = high_1d[0] - close_1d[0]  # Approximation for first bar
    tr3[0] = low_1d[0] - close_1d[0]   # Approximation for first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate breakout levels: 1d high/low ± 0.5*ATR(10)
    breakout_up = high_1d + 0.5 * atr_10
    breakout_down = low_1d - 0.5 * atr_10
    
    # Align breakout levels to 12h timeframe (wait for completed daily bar)
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_down_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d high + 0.5*ATR(10), above 12h EMA50, volume confirmation, in session
            if (close[i] > breakout_up_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d low - 0.5*ATR(10), below 12h EMA50, volume confirmation, in session
            elif (close[i] < breakout_down_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 OR volume drops below average
            if close[i] < ema50_12h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 OR volume drops below average
            if close[i] > ema50_12h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals