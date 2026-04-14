#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for price channel (12h timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12-period EMA on 1d close for trend direction
    close_1d_series = pd.Series(close_1d)
    ema_12_1d = close_1d_series.ewm(span=12, adjust=False).values
    
    # Calculate ATR on 1d for volatility filter
    tr = np.maximum(high_1d - low_1d,
                    np.maximum(np.abs(high_1d - np.roll(high_1d, 1)),
                               np.abs(low_1d - np.roll(low_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    ema_12_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_12_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume ratio: current 12h volume vs 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    vol_ratio = np.where(vol_ma_20 > 0, volume / vol_ma_20, 0)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size to limit trades and manage risk
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(ema_12_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 1d high + volume surge + uptrend (EMA up)
            if (close[i] > high_1d_aligned[i] and
                vol_ratio[i] > 2.0 and
                ema_12_1d_aligned[i] > ema_12_1d_aligned[i-1]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 1d low + volume surge + downtrend (EMA down)
            elif (close[i] < low_1d_aligned[i] and
                  vol_ratio[i] > 2.0 and
                  ema_12_1d_aligned[i] < ema_12_1d_aligned[i-1]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below 1d low or trend reverses
            if (close[i] < low_1d_aligned[i] or
                ema_12_1d_aligned[i] < ema_12_1d_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above 1d high or trend reverses
            if (close[i] > high_1d_aligned[i] or
                ema_12_1d_aligned[i] > ema_12_1d_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_EMA_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0