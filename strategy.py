#!/usr/bin/env python3
"""
1D_KAMA_Trend_Filter_With_RSI_and_Chop
Hypothesis: 1d KAMA identifies trend direction, RSI filters overbought/oversold, and Choppiness index avoids ranging markets.
In trending markets (CHOP < 38.2), enter long when KAMA rising and RSI > 50, short when KAMA falling and RSI < 50.
Exit when trend weakens (KAMA flips) or market becomes choppy (CHOP > 61.8).
Weekly trend filter ensures alignment with higher timeframe momentum.
Designed for low turnover (<15 trades/year) to minimize fee drag in 2025 bear market.
"""

name = "1D_KAMA_Trend_Filter_With_RSI_and_Chop"
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
    
    # ===== KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA) =====
    # Fast = 2, Slow = 30
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constant
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # Seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ===== RSI (14) =====
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=np.float64)
    avg_loss = np.full_like(close, np.nan, dtype=np.float64)
    
    # First average (simple)
    avg_gain[13] = np.mean(gain[0:14])
    avg_loss[13] = np.mean(loss[0:14])
    
    # Wilder smoothing
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # ===== CHOPPINESS INDEX (14) =====
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align with close
    
    # ATR (14)
    atr = np.full_like(close, np.nan, dtype=np.float64)
    atr[13] = np.nanmean(tr[1:15])  # First ATR
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of TR over 14 periods
    sum_tr = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(13, n):
        if i == 13:
            sum_tr[i] = np.sum(tr[1:15])
        else:
            sum_tr[i] = sum_tr[i-1] - tr[i-13] + tr[i]
    
    # Max and min close over 14 periods
    max_close = np.full_like(close, np.nan, dtype=np.float64)
    min_close = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(13, n):
        max_close[i] = np.max(high[i-13:i+1])
        min_close[i] = np.min(low[i-13:i+1])
    
    # Chop calculation
    chop = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(13, n):
        if sum_tr[i] > 0 and max_close[i] > min_close[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (max_close[i] - min_close[i])) / np.log10(14)
    
    # ===== WEEKLY TREND FILTER =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ===== ALIGN HTF INDICATORS TO DAILY =====
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    chop_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), chop)
    
    # ===== SIGNAL PARAMETERS =====
    position_size = 0.25  # Conservative size to manage drawdown
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any data invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        trending_market = chop_aligned[i] < 38.2
        choppy_market = chop_aligned[i] > 61.8
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + trending market + weekly uptrend
            if kama_rising and rsi_above_50 and trending_market and weekly_uptrend:
                signals[i] = position_size
                position = 1
            # Short: KAMA falling + RSI < 50 + trending market + weekly downtrend
            elif kama_falling and rsi_below_50 and trending_market and weekly_downtrend:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: KAMA falls OR market choppy OR weekly trend turns down
                if (not kama_rising) or choppy_market or (not weekly_uptrend):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: KAMA rises OR market choppy OR weekly trend turns up
                if (not kama_falling) or choppy_market or (not weekly_downtrend):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals