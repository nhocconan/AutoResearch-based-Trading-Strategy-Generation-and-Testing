#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily timeframe with weekly KAMA trend + daily RSI reversal + volume confirmation.
# Uses weekly KAMA for trend direction, daily RSI for mean-reversion entries.
# Enters long when RSI < 30 in bullish weekly trend, short when RSI > 70 in bearish weekly trend.
# Volume filter ensures institutional participation. Designed for 15-25 trades/year.
# Weekly trend filter reduces whipsaw in sideways markets and improves win rate in both bull/bear.

name = "1d_1w_kama_rsi_rev_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA (Kaufman Adaptive Moving Average)
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, k=1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1w, np.nan)
    kama[29] = close_1w[29]  # seed
    for i in range(30, len(close_1w)):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close_1w[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Weekly trend: price above KAMA = bullish, below = bearish
    weekly_trend_bull = close_1w > kama
    weekly_trend_bear = close_1w < kama
    
    # Align weekly trend to daily
    weekly_trend_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bull)
    weekly_trend_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bear)
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    # Wilder smoothing
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily average volume (20-period)
    vol_avg_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(vol_avg_20[i]) or
            np.isnan(weekly_trend_bull_aligned[i]) or np.isnan(weekly_trend_bear_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * 20-day average volume
        vol_filter = volume[i] > 1.3 * vol_avg_20[i]
        
        # Determine weekly trend direction
        is_bullish_week = weekly_trend_bull_aligned[i]
        is_bearish_week = weekly_trend_bear_aligned[i]
        
        # RSI reversal signals
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Enter long: RSI oversold in bullish weekly trend
        enter_long = rsi_oversold and vol_filter and is_bullish_week
        # Enter short: RSI overbought in bearish weekly trend
        enter_short = rsi_overbought and vol_filter and is_bearish_week
        
        # Exit when RSI returns to neutral zone (40-60) or opposite extreme
        exit_long = position == 1 and (rsi[i] >= 40 or rsi[i] > 60)
        exit_short = position == -1 and (rsi[i] <= 60 or rsi[i] < 40)
        
        # Priority: entry > exit > hold
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals