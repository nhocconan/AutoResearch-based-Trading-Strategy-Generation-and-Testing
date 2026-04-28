#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Filter
Hypothesis: Daily KAMA direction with RSI filter and volume confirmation. 
KAMA adapts to market noise, reducing whipsaw in choppy markets. 
Trades only when RSI confirms momentum (>55 for long, <45 for short) and volume surges.
Works in both bull and bear markets by following adaptive trend while filtering false signals.
Target: 15-25 trades/year.
"""

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
    
    # Get weekly data for trend filter (optional but adds robustness)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Parameters: ER fast = 2, slow = 30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    price_change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility_sum = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility_sum[i] = volatility_sum[i-1] + np.abs(close[i] - close[i-1])
    # For first 10 periods, use expanding window
    volatility_10 = np.zeros_like(close)
    for i in range(len(close)):
        if i < 10:
            volatility_10[i] = np.sum(np.abs(np.diff(close[:i+1]))) if i > 0 else 0
        else:
            volatility_10[i] = volatility_sum[i] - volatility_sum[i-10]
    
    ER = np.where(volatility_10 > 0, price_change / volatility_10, 0)
    # Smooth ER
    fast_SC = 2 / (2 + 1)   # EMA 2
    slow_SC = 2 / (30 + 1)  # EMA 30
    SC = (ER * (fast_SC - slow_SC) + slow_SC) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + SC[i] * (close[i] - kama[i-1])
    
    # Alternative simpler approach using pandas for robustness
    # Recalculate using pandas Series for clarity
    close_series = pd.Series(close)
    change = close_series.diff(10).abs()
    volatility = close_series.diff().abs().rolling(10).sum()
    ER = change / volatility
    ER = ER.fillna(0)
    fast_SC = 2 / (2 + 1)
    slow_SC = 2 / (30 + 1)
    SC = (ER * (fast_SC - slow_SC) + slow_SC) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + SC.iloc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    # Align higher timeframe data
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA
        trend_up = close > ema_34_1w_aligned
        trend_down = close < ema_34_1w_aligned
        
        # Entry conditions
        # Long: price > KAMA AND RSI > 55 (bullish momentum) AND volume surge AND weekly uptrend
        long_entry = (close[i] > kama_aligned[i] and 
                     rsi[i] > 55 and 
                     volume_surge[i] and 
                     trend_up[i])
        
        # Short: price < KAMA AND RSI < 45 (bearish momentum) AND volume surge AND weekly downtrend
        short_entry = (close[i] < kama_aligned[i] and 
                      rsi[i] < 45 and 
                      volume_surge[i] and 
                      trend_down[i])
        
        # Exit when price crosses KAMA in opposite direction
        long_exit = close[i] < kama_aligned[i] and position == 1
        short_exit = close[i] > kama_aligned[i] and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Close long
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.25   # Close short
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

name = "1d_KAMA_Trend_RSI_Filter"
timeframe = "1d"
leverage = 1.0