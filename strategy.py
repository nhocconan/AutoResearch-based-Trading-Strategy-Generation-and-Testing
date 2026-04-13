#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for session filter
    hours = prices.index.hour
    
    # Daily data for ATR and close (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first bar
    tr3[0] = tr1[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily close EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 1h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to daily EMA50
        long_trend = close[i] > ema_50_1d_aligned[i]
        short_trend = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: current 1h ATR (approximated from daily ATR/24) 
        # Used for dynamic threshold
        atr_1h_approx = atr_14_aligned[i] / 24  # rough approximation
        
        # Entry conditions: volatility breakout with trend filter
        # Long when price breaks above close + 0.5*ATR with uptrend
        # Short when price breaks below close - 0.5*ATR with downtrend
        entry_long = close[i] > close_1d[-1] + 0.5 * atr_1h_approx if i > 0 else False
        entry_short = close[i] < close_1d[-1] - 0.5 * atr_1h_approx if i > 0 else False
        
        # Use previous close for comparison (available at bar i)
        prev_close = close[i-1]
        entry_long = close[i] > prev_close + 0.5 * atr_1h_approx
        entry_short = close[i] < prev_close - 0.5 * atr_1h_approx
        
        if position == 0:
            if entry_long and long_trend:
                position = 1
                signals[i] = position_size
            elif entry_short and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price closes below EMA50 (trend change)
            if close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price closes above EMA50 (trend change)
            if close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_1d_ATR_Volatility_Breakout_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0