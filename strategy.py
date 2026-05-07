#!/usr/bin/env python3
name = "1d_Fibonacci_Retracement_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly high/low over last 52 weeks (1 year)
    lookback_weeks = min(52, len(df_1w))
    weekly_highs = df_1w['high'].values[-lookback_weeks:]
    weekly_lows = df_1w['low'].values[-lookback_weeks:]
    
    # Find highest high and lowest low in the lookback period
    period_high = np.max(weekly_highs)
    period_low = np.min(weekly_lows)
    fib_range = period_high - period_low
    
    # Calculate Fibonacci levels
    fib_236 = period_high - 0.236 * fib_range
    fib_382 = period_high - 0.382 * fib_range
    fib_618 = period_high - 0.618 * fib_range
    fib_786 = period_high - 0.786 * fib_range
    
    # Align Fibonacci levels to daily timeframe
    fib_236_arr = np.full(len(df_1w), fib_236)
    fib_382_arr = np.full(len(df_1w), fib_382)
    fib_618_arr = np.full(len(df_1w), fib_618)
    fib_786_arr = np.full(len(df_1w), fib_786)
    
    fib_236_aligned = align_htf_to_ltf(prices, df_1w, fib_236_arr)
    fib_382_aligned = align_htf_to_ltf(prices, df_1w, fib_382_arr)
    fib_618_aligned = align_htf_to_ltf(prices, df_1w, fib_618_arr)
    fib_786_aligned = align_htf_to_ltf(prices, df_1w, fib_786_arr)
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily indicators
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(fib_236_aligned[i]) or np.isnan(fib_382_aligned[i]) or 
            np.isnan(fib_618_aligned[i]) or np.isnan(fib_786_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        # Weekly trend condition
        weekly_uptrend = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
        weekly_downtrend = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]
        
        if position == 0:
            # Long: Price pulls back to Fibonacci support in weekly uptrend
            if (weekly_uptrend and vol_condition and 
                close[i] <= fib_618_aligned[i] * 1.01 and close[i] >= fib_618_aligned[i] * 0.99 and
                rsi_values[i] < 40):
                signals[i] = 0.25
                position = 1
            # Short: Price retraces to Fibonacci resistance in weekly downtrend
            elif (weekly_downtrend and vol_condition and 
                  close[i] >= fib_382_aligned[i] * 0.99 and close[i] <= fib_382_aligned[i] * 1.01 and
                  rsi_values[i] > 60):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price reaches Fibonacci extension or RSI overbought
            if (close[i] >= fib_236_aligned[i] * 0.99 or 
                rsi_values[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price reaches Fibonacci extension or RSI oversold
            if (close[i] <= fib_618_aligned[i] * 1.01 or 
                rsi_values[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Fibonacci retracement levels on weekly charts provide strong support/resistance
# - In weekly uptrends, price often pulls back to 61.8% Fibonacci level before continuing up
# - In weekly downtrends, price often retraces to 38.2% Fibonacci level before continuing down
# - Weekly EMA20 trend filter ensures we only trade in the direction of the higher timeframe trend
# - RSI (14) filters for oversold/overbought conditions at Fibonacci levels
# - Volume confirmation (1.5x average) reduces false signals
# - Position size 0.25 targets ~20-60 trades/year to avoid fee drag
# - Works in both bull and bear markets by trading with the weekly trend
# - Fibonacci levels are widely watched and act as self-fulfilling prophecy
# - Aims for 40-120 total trades over 4 years (10-30/year) to stay within limits