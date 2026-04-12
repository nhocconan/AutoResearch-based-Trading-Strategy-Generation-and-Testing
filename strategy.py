#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_channel_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Keltner Channel and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily ATR for Keltner Channel
    atr_1d = pd.Series(df_1d['high'] - df_1d['low']).rolling(window=14, min_periods=14).mean().values
    # Daily EMA for middle line
    ema_20 = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Keltner Channel bands
    upper = ema_20 + 2 * atr_1d
    lower = ema_20 - 2 * atr_1d
    
    # Map to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume confirmation: current 4h volume > 20-period average of daily volume
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
    vol_ma = pd.Series(vol_1d_aligned).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above upper Keltner band with volume
        long_signal = (close[i] > upper_aligned[i] and volume_filter[i])
        
        # Short: price breaks below lower Keltner band with volume
        short_signal = (close[i] < lower_aligned[i] and volume_filter[i])
        
        # Exit: price returns to EMA(20)
        exit_long = (position == 1 and close[i] < ema_20_aligned[i])
        exit_short = (position == -1 and close[i] > ema_20_aligned[i])
        
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