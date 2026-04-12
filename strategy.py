#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
    # Buy when price breaks above Camarilla H3 level in uptrend (1d EMA50)
    # Sell when price breaks below Camarilla L3 level in downtrend
    # Volume > 1.3x 20-period MA confirms breakout strength
    # Discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (need previous day's OHLC)
    # We'll use the 1d data shifted by 1 to get previous completed day
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev = np.roll(df_1d['high'].values, 1)
    low_1d_prev = np.roll(df_1d['low'].values, 1)
    
    # First bar: no previous day data
    close_1d_prev[0] = close_1d[0]
    high_1d_prev[0] = df_1d['high'].values[0]
    low_1d_prev[0] = df_1d['low'].values[0]
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_h3[i] = close_1d_prev[i]  # fallback
            camarilla_l3[i] = close_1d_prev[i]
        else:
            daily_range = high_1d_prev[i] - low_1d_prev[i]
            camarilla_h3[i] = close_1d_prev[i] + daily_range * 1.1 / 4
            camarilla_l3[i] = close_1d_prev[i] - daily_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA(50)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        long_entry = long_breakout and (vol_ratio[i] > 1.3) and uptrend
        short_entry = short_breakout and (vol_ratio[i] > 1.3) and downtrend
        
        # Exit conditions: price returns to previous day's close (mean reversion)
        long_exit = close[i] < camarilla_h3_aligned[i]  # price back below H3
        short_exit = close[i] > camarilla_l3_aligned[i]  # price back above L3
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_vol_trend_v1"
timeframe = "4h"
leverage = 1.0