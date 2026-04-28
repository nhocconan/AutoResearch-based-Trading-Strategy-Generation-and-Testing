#!/usr/bin/env python3
"""
1d_WeeklyCandle_Pullback_Momentum
Hypothesis: On daily timeframe, enter long after a pullback to the 20-day EMA in a weekly uptrend (weekly close > weekly open), and short after a pullback in a weekly downtrend (weekly close < weekly open). Use RSI(2) for mean-reversion entry timing and volume confirmation to filter weak moves. Designed for low trade frequency (~10-20/year) to avoid fee drag while capturing medium-term swings in both bull and bear markets.
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Weekly trend: bullish when weekly close > weekly open
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align weekly trend to daily timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bearish.astype(float))
    
    # Daily 20 EMA for pullback
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # RSI(2) for mean-reversion entry
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi2 = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(ema20[i]) or np.isnan(rsi2[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Pullback conditions: price near 20 EMA (within 1%)
        near_ema = abs(close[i] - ema20[i]) / ema20[i] < 0.01
        
        # RSI conditions for mean-reversion entry
        rsi_oversold = rsi2[i] < 15
        rsi_overbought = rsi2[i] > 85
        
        # Entry conditions
        long_entry = near_ema and rsi_oversold and weekly_bullish_aligned[i] and volume_surge[i]
        short_entry = near_ema and rsi_overbought and weekly_bearish_aligned[i] and volume_surge[i]
        
        # Exit on opposite signal or RSI normalization
        long_exit = rsi2[i] > 60
        short_exit = rsi2[i] < 40
        
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

name = "1d_WeeklyCandle_Pullback_Momentum"
timeframe = "1d"
leverage = 1.0