# %%
#!/usr/bin/env python3
"""
[40564] 1d_1w_kama_rsi_chop
Hypothesis: Daily KAMA trend direction with RSI momentum filter and weekly Choppiness Index regime filter.
Uses KAMA (adaptive moving average) to identify trend direction, RSI for momentum strength,
and weekly Choppiness Index to filter trades to trending regimes only (CHOP < 38.2).
Designed to work in both bull and bear markets by focusing on strong trending moves.
Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
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
    
    # Get weekly data for Choppiness Index (trend regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Choppiness Index (14-period)
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # True Range calculation
    tr1 = wh[1:] - wl[1:]
    tr2 = np.abs(wh[1:] - wc[:-1])
    tr3 = np.abs(wl[1:] - wc[:-1])
    tr_first = np.max([wh[0] - wl[0], np.abs(wh[0] - wc[0]), np.abs(wl[0] - wc[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(wh).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(wl).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Avoid division by zero
    chop = np.where(range_max_min != 0, 
                    100 * np.log10(atr_sum / range_max_min) / np.log10(14), 
                    50)  # neutral when no range
    
    # Market regime: CHOP > 61.8 = range, CHOP < 38.2 = trend
    trending_weekly = chop < 38.2
    trending_weekly_aligned = align_htf_to_ltf(prices, df_1w, trending_weekly.astype(float))
    
    # Calculate daily KAMA (adaptive moving average)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.abs(np.diff(close, n=1))  # |close - close[1]|
    
    # Pad arrays for calculation
    change_padded = np.concatenate([[np.nan] * 10, change])
    volatility_padded = np.concatenate([[np.nan], volatility])
    
    # Calculate ER using rolling sum
    change_sum = pd.Series(change_padded).rolling(window=10, min_periods=10).sum().values[10:]
    volatility_sum = pd.Series(volatility_padded).rolling(window=10, min_periods=10).sum().values
    
    # ER = change_sum / volatility_sum, handle division by zero
    er = np.where(volatility_sum != 0, change_sum / volatility_sum, 0)
    
    # Smoothing constants: sc = [ER * (fastest - slowest) + slowest]^2
    # fastest = 2/(2+1) = 0.6667, slowest = 2/(30+1) = 0.0645
    sc = (er * 0.6022 + 0.0645) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    
    for i in range(10, n):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_aligned = kama  # Already daily
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    # Handle first average
    avg_gain = np.concatenate([[np.nan] * 13, avg_gain])
    avg_loss = np.concatenate([[np.nan] * 13, avg_loss])
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(trending_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Price above/below KAMA + RSI momentum + weekly trend
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        rsi_overbought = rsi[i] > 60  # Momentum confirmation for longs
        rsi_oversold = rsi[i] < 40   # Momentum confirmation for shorts
        weekly_trend = trending_weekly_aligned[i] > 0.5
        
        long_entry = price_above_kama and rsi_overbought and weekly_trend
        short_entry = price_below_kama and rsi_oversold and weekly_trend
        
        # Exit when price crosses KAMA in opposite direction
        exit_long = position == 1 and close[i] < kama_aligned[i]
        exit_short = position == -1 and close[i] > kama_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_chop"
timeframe = "1d"
leverage = 1.0
# %%