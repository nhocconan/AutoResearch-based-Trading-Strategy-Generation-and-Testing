#!/usr/bin/env python3
"""
4h_1d_KAMA_RSI_Trend_v1
Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) trend direction combined with RSI momentum and 1-day Bollinger Bands for mean reversion signals. Designed to capture trending moves in bull markets and mean-reversion bounces in bear markets with tight entry conditions to limit trade frequency (<30 trades/year per symbol).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) for trend
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder for correct calc
    # Correct calculation:
    change = np.abs(np.diff(close, prepend=close[0]))
    # Volatility is sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(len(close)):
        if i < 10:
            volatility[i] = np.nan
        else:
            volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    # Avoid loop by using rolling sum of absolute differences
    diff_abs = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(diff_abs).rolling(window=10, min_periods=10).sum().values
    # ER = change / volatility, handle division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # SC = [ER * (fastest - slowest) + slowest]^2, fastest=2/(2+1), slowest=2/(30+1)
    fastest = 2 / (2 + 1)
    slowest = 2 / (30 + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-day Bollinger Bands (20-period, 2 std)
    close_1d = df_1d['close'].values
    bb_mid_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper_1d = bb_mid_1d + 2 * bb_std_1d
    bb_lower_1d = bb_mid_1d - 2 * bb_std_1d
    
    # Align 1d indicators to 4h timeframe
    bb_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_upper_1d)
    bb_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_lower_1d)
    bb_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_mid_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(bb_upper_1d_aligned[i]) or np.isnan(bb_lower_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below KAMA
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # Mean reversion signals from Bollinger Bands
        near_lower_bb = close[i] <= bb_lower_1d_aligned[i] * 1.01  # within 1% of lower band
        near_upper_bb = close[i] >= bb_upper_1d_aligned[i] * 0.99  # within 1% of upper band
        
        # RSI conditions: oversold (<30) or overbought (>70)
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Entry conditions
        long_entry = near_lower_bb and rsi_oversold and uptrend
        short_entry = near_upper_bb and rsi_overbought and downtrend
        
        # Exit conditions: return to middle band or trend reversal
        long_exit = (close[i] >= bb_mid_1d_aligned[i]) or (not uptrend)
        short_exit = (close[i] <= bb_mid_1d_aligned[i]) or (not downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals