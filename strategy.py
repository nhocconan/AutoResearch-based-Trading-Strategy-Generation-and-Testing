#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_pullback_v1"
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
    
    # Get daily data for CCI and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily CCI(20)
    tp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    ma_tp = tp_1d.rolling(window=20, min_periods=20).mean()
    md_tp = tp_1d.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_1d = (tp_1d - ma_tp) / (0.015 * md_tp)
    cci_1d_values = cci_1d.values
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d_values)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h volume filter: current volume > 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(cci_1d_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: CCI < -100 (oversold) + price > EMA50 (uptrend) + volume
        long_signal = (cci_1d_aligned[i] < -100 and close[i] > ema_50_aligned[i] and volume_filter[i])
        
        # Short: CCI > 100 (overbought) + price < EMA50 (downtrend) + volume
        short_signal = (cci_1d_aligned[i] > 100 and close[i] < ema_50_aligned[i] and volume_filter[i])
        
        # Exit: CCI returns to neutral zone (-50 to 50)
        exit_long = (position == 1 and cci_1d_aligned[i] > -50)
        exit_short = (position == -1 and cci_1d_aligned[i] < 50)
        
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