#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h/1d confluence for direction and 1h for timing.
    # Long when: price > 4h EMA50 AND price > 1d EMA200 AND RSI(14) > 50 (bullish bias)
    # Short when: price < 4h EMA50 AND price < 1d EMA200 AND RSI(14) < 50 (bearish bias)
    # Exit: reverse signal or RSI extremes (RSI>70 for long exit, RSI<30 for short exit)
    # Session filter: 08-20 UTC to reduce noise.
    # Discrete sizing: 0.20 to limit drawdown and fee churn.
    # Target: 15-37 trades/year (60-150 over 4 years) via strict HTF alignment.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for longer-term trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Calculate 1d EMA(200) for long-term trend
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate 1h RSI(14) for momentum and exit signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[1:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if HTF data not ready
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(ema_1d_200_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Trend alignment: both 4h and 1d EMAs must agree
        bullish_align = (close[i] > ema_4h_50_aligned[i]) and (close[i] > ema_1d_200_aligned[i])
        bearish_align = (close[i] < ema_4h_50_aligned[i]) and (close[i] < ema_1d_200_aligned[i])
        
        # RSI conditions
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        rsi_overbought = rsi[i] > 70  # Exit long
        rsi_oversold = rsi[i] < 30    # Exit short
        
        # Entry logic
        long_entry = bullish_align and rsi_bullish
        short_entry = bearish_align and rsi_bearish
        
        # Exit logic
        long_exit = not bullish_align or rsi_overbought
        short_exit = not bearish_align or rsi_oversold
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_ema_rsi_confluence_session_v1"
timeframe = "1h"
leverage = 1.0