#!/usr/bin/env python3
"""
1d_Fibonacci_Retracement_Momentum_Extension_v1
Hypothesis: Daily Fibonacci retracement levels (61.8%) from weekly swing high/low act as strong support/resistance. 
Long when price pulls back to 61.8% retracement in a weekly uptrend with momentum confirmation; short when price retraces to 61.8% extension in a weekly downtrend.
Uses weekly trend filter to avoid counter-trend trades and momentum oscillator (RSI) for entry timing, targeting 15-25 trades/year to minimize fee drag.
Works in bull markets via pullbacks to support and in bear markets via bounces from resistance.
"""

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
    
    # Get weekly data for trend and swing points
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly trend: EMA50
    weekly_close = df_weekly['close'].values
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = ema50_weekly > np.roll(ema50_weekly, 1)
    weekly_downtrend = ema50_weekly < np.roll(ema50_weekly, 1)
    weekly_uptrend[0] = False
    weekly_downtrend[0] = False
    
    # Weekly swing points (highest high and lowest low over last 12 weeks ~ 3 months)
    lookback = 12
    highest_high = pd.Series(df_weekly['high']).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(df_weekly['low']).rolling(window=lookback, min_periods=lookback).min().values
    
    # Fibonacci levels: 61.8% retracement and extension
    range_weekly = highest_high - lowest_low
    fib_618_retracement = lowest_low + 0.618 * range_weekly  # Support in uptrend
    fib_618_extension = highest_high - 0.618 * range_weekly  # Resistance in downtrend
    
    # Align weekly data to daily
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_downtrend.astype(float))
    fib_618_retracement_aligned = align_htf_to_ltf(prices, df_weekly, fib_618_retracement)
    fib_618_extension_aligned = align_htf_to_ltf(prices, df_weekly, fib_618_extension)
    
    # Daily momentum: RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # RSI conditions: not overbought/oversold for better entry
    rsi_not_overbought = rsi < 70
    rsi_not_oversold = rsi > 30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i]) or np.isnan(fib_618_retracement_aligned[i]) or
            np.isnan(fib_618_extension_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Long: weekly uptrend, price at 61.8% retracement support, RSI not overbought
        long_entry = (weekly_uptrend_aligned[i] > 0.5 and 
                     abs(close[i] - fib_618_retracement_aligned[i]) < (0.005 * close[i]) and  # Within 0.5%
                     rsi_not_overbought[i])
        
        # Short: weekly downtrend, price at 61.8% extension resistance, RSI not oversold
        short_entry = (weekly_downtrend_aligned[i] > 0.5 and 
                      abs(close[i] - fib_618_extension_aligned[i]) < (0.005 * close[i]) and  # Within 0.5%
                      rsi_not_oversold[i])
        
        # Exit: trend reversal or price moves 2% away from level
        long_exit = (weekly_uptrend_aligned[i] < 0.5 or 
                    close[i] < fib_618_retracement_aligned[i] * 0.98)
        short_exit = (weekly_downtrend_aligned[i] < 0.5 or 
                     close[i] > fib_618_extension_aligned[i] * 1.02)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Fibonacci_Retracement_Momentum_Extension_v1"
timeframe = "1d"
leverage = 1.0