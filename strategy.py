#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: current volume > 1.5x average volume (conservative to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > vol_ma * 1.5
    
    # 12-period high/low for breakout levels
    high_12 = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_12 = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 30, 12)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or
            np.isnan(high_12[i]) or
            np.isnan(low_12[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: ATR > 0.5 * price (avoid low volatility choppy periods)
        vol_filter = atr_14_aligned[i] > 0.005 * close[i]
        
        # Breakout conditions with volatility and volume confirmation
        long_breakout = close[i] > high_12[i-1]  # Break above 12-period high
        short_breakout = close[i] < low_12[i-1]  # Break below 12-period low
        
        long_entry = long_breakout and uptrend and vol_filter and volume_confirm[i]
        short_entry = short_breakout and downtrend and vol_filter and volume_confirm[i]
        
        # Exit conditions: when trend reverses or volatility drops
        if position == 1:
            exit_condition = not (uptrend and vol_filter)
        elif position == -1:
            exit_condition = not (downtrend and vol_filter)
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_EMA34_Trend_ATRVol_VolumeConfirm_Breakout"
timeframe = "12h"
leverage = 1.0