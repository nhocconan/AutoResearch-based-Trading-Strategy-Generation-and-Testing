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
    
    # Load 1d data for calculations (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's high/low/close
    prev_high = high_1d
    prev_low = low_1d
    prev_close = close_1d
    pivot_range = prev_high - prev_low
    
    # Calculate 1d RSI(14) for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # First bar TR
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily volatility filter: ATR ratio
    atr_ratio = atr_1d / np.roll(atr_1d, 5)  # Current ATR / 5-day ago ATR
    
    # Calculate 1-day change for momentum
    daily_change = (close_1d / np.roll(close_1d, 1) - 1) * 100
    daily_change[0] = 0  # First bar
    
    # Align all 1d indicators to 1d timeframe (we're already on 1d timeframe)
    rsi_1d_aligned = rsi_1d_values
    ema_50_1d_aligned = ema_50_1d
    atr_ratio_aligned = atr_ratio
    daily_change_aligned = daily_change
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(daily_change_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. RSI < 30 (oversold)
            # 2. Price > EMA50 (uptrend filter)
            # 3. ATR ratio > 1.2 (volatility expansion)
            # 4. Daily change < -1% (recent weakness)
            if (rsi_1d_aligned[i] < 30 and 
                close[i] > ema_50_1d_aligned[i] and 
                atr_ratio_aligned[i] > 1.2 and 
                daily_change_aligned[i] < -1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. RSI > 70 (overbought)
            # 2. Price < EMA50 (downtrend filter)
            # 3. ATR ratio > 1.2 (volatility expansion)
            # 4. Daily change > 1% (recent strength)
            elif (rsi_1d_aligned[i] > 70 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  atr_ratio_aligned[i] > 1.2 and 
                  daily_change_aligned[i] > 1.0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions:
            # Long exit: RSI > 50 (overbought) OR price < EMA50
            # Short exit: RSI < 50 (oversold) OR price > EMA50
            if position == 1:
                if rsi_1d_aligned[i] > 50 or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi_1d_aligned[i] < 50 or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_RSI_MeanReversion_Volatility_Trend_Filter"
timeframe = "1d"
leverage = 1.0