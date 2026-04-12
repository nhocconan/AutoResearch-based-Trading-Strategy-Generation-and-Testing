#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 5 or len(df_1w) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla levels (H4/L4)
    camarilla_high_1d = np.zeros(len(close_1d))
    camarilla_low_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        camarilla_high_1d[i] = C + ((H - L) * 1.1 / 2)
        camarilla_low_1d[i] = C - ((H - L) * 1.1 / 2)
    
    # Calculate 1w weekly pivot (for trend direction)
    weekly_pivot = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        H = high_1w[i-1]
        L = low_1w[i-1]
        C = close_1w[i-1]
        weekly_pivot[i] = (H + L + C) / 3
    
    # Align to 6h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high_1d)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low_1d)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume filter: current volume > 20-period average (on 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Momentum filter: 6h close > 6h EMA(20)
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    momentum_up = close > ema_20
    momentum_down = close < ema_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # warmup for volume/EMA
        # Skip if not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_high_aligned[i]
        breakout_down = close[i] < camarilla_low_aligned[i]
        
        # Additional filters
        vol_ok = volume_ok[i]
        mom_up = momentum_up[i]
        mom_down = momentum_down[i]
        
        # Trend filter: use weekly pivot
        above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Entry signals with all filters
        long_signal = breakout_up and vol_ok and mom_up and above_weekly_pivot
        short_signal = breakout_down and vol_ok and mom_down and below_weekly_pivot
        
        # Exit when price returns to the 1d Camarilla pivot (H4/L4 levels act as support/resistance)
        exit_long = close[i] < camarilla_low_aligned[i]  # fallback to support
        exit_short = close[i] > camarilla_high_aligned[i]  # fallback to resistance
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals