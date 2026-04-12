#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 21-period EMA on 1w data
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 9 and 21 period EMA on 1d data
    close_series = pd.Series(close)
    ema_9 = close_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter - 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_9[i]) or 
            np.isnan(ema_21[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below weekly EMA21
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # EMA cross signals
        ema_cross_up = ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]
        ema_cross_down = ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]
        
        # Long: EMA cross up in uptrend with volume
        long_signal = ema_cross_up and uptrend and volume_ok[i]
        # Short: EMA cross down in downtrend with volume
        short_signal = ema_cross_down and downtrend and volume_ok[i]
        
        # Exit on opposite EMA cross
        exit_long = ema_cross_down
        exit_short = ema_cross_up
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals