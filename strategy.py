#!/usr/bin/env python3
"""
12h_1d_RSI_Streak_Reversal_Strategy
Hypothesis: RSI streaks on daily timeframe identify overextended moves. 
Three consecutive RSI closes above 70 (overbought) or below 30 (oversold) 
signal reversals. Enter on close of third candle with volume confirmation 
and 12h trend filter (price vs 20-period EMA). Exit on RSI mean reversion 
(to 50) or opposite streak. Designed for low frequency (15-25 trades/year) 
to minimize fee drag while capturing mean reversion in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RSI_Streak_Reversal_Strategy"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY RSI CALCULATION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = np.nan  # Not enough data
    
    # === RSI STREAK DETECTION ===
    # Streak of 3+ consecutive closes above 70 (overbought) or below 30 (oversold)
    rsi_above_70 = rsi > 70
    rsi_below_30 = rsi < 30
    
    # Count consecutive days
    streak_above = np.zeros_like(rsi)
    streak_below = np.zeros_like(rsi)
    
    for i in range(1, len(rsi)):
        if rsi_above_70[i]:
            streak_above[i] = streak_above[i-1] + 1
        else:
            streak_above[i] = 0
            
        if rsi_below_30[i]:
            streak_below[i] = streak_below[i-1] + 1
        else:
            streak_below[i] = 0
    
    # Signal when streak reaches 3
    overbought_signal = streak_above >= 3
    oversold_signal = streak_below >= 3
    
    # Align to 12h timeframe
    overbought_aligned = align_htf_to_ltf(prices, df_1d, overbought_signal.astype(float))
    oversold_aligned = align_htf_to_ltf(prices, df_1d, oversold_signal.astype(float))
    
    # === 12h TREND FILTER (EMA 20) ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(overbought_aligned[i]) or np.isnan(oversold_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        # Long: RSI streak oversold (3+ days <30) + volume + price above EMA20
        long_signal = (oversold_aligned[i] > 0.5) and (vol_ratio[i] > 1.5) and (close[i] > ema20[i])
        
        # Short: RSI streak overbought (3+ days >70) + volume + price below EMA20
        short_signal = (overbought_aligned[i] > 0.5) and (vol_ratio[i] > 1.5) and (close[i] < ema20[i])
        
        # Exit: RSI returns to neutral zone (40-60) or opposite streak
        # Need current day's RSI for exit (use previous day's aligned value)
        rsi_prev = align_htf_to_ltf(prices, df_1d, rsi)
        exit_long = (rsi_prev[i] > 40) and (rsi_prev[i] < 60) and position == 1
        exit_short = (rsi_prev[i] > 40) and (rsi_prev[i] < 60) and position == -1
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals