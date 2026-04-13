#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for ATR and range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily ATR for volatility-based stops and sizing
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily range for breakout levels
    daily_range = df_1d['high'] - df_1d['low']
    
    # Weekly EMA for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Align all data to 1h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w.values)
    
    # Hourly ATR for entry timing
    tr_h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_h[0] = high[0] - low[0]  # First bar
    atr_14_h = pd.Series(tr_h).rolling(window=14, min_periods=14).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(daily_range_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Trend filter: only long when price > weekly EMA50, short when price < weekly EMA50
        long_trend = close[i] > ema_50_1w_aligned[i]
        short_trend = close[i] < ema_50_1w_aligned[i]
        
        # Breakout condition: price breaks daily high/low with volatility expansion
        # Long breakout: price > daily high + 0.5 * ATR
        # Short breakdown: price < daily low - 0.5 * ATR
        long_breakout = close[i] > high[i-1] + 0.5 * atr_14_1d_aligned[i]
        short_breakout = close[i] < low[i-1] - 0.5 * atr_14_1d_aligned[i]
        
        # Volume confirmation: current hour volume > 1.5x 14-period average
        vol_condition = volume[i] > (atr_14_h[i] * 1.5)  # Using ATR as proxy for normal volatility
        
        if position == 0:
            if long_breakout and vol_condition and long_trend:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reverses or volatility contracts
            if close[i] < ema_50_1w_aligned[i] or atr_14_h[i] < 0.5 * atr_14_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reverses or volatility contracts
            if close[i] > ema_50_1w_aligned[i] or atr_14_h[i] < 0.5 * atr_14_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_1d1w_ATR_Breakout_With_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0