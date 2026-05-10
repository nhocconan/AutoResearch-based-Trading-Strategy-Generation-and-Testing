#!/usr/bin/env python3
"""
1d_KAMA_Trend_with_RSI_and_Chop
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) to capture trend direction on daily timeframe,
combined with RSI for momentum confirmation and Choppiness Index to filter ranging markets.
Enter long when KAMA slopes up, RSI > 50, and CHOP > 61.8 (ranging market - mean reversion setup).
Enter short when KAMA slopes down, RSI < 50, and CHOP > 61.8.
Exit when conditions reverse. Uses weekly trend filter to ensure alignment with higher timeframe trend.
Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe.
Works in bull markets via trend following and in bear markets via mean reversion in ranging regimes.
"""

name = "1d_KAMA_Trend_with_RSI_and_Chop"
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
    
    # --- 1d Indicators ---
    # KAMA (Kaufman Adaptive Moving Average)
    def kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama_val = np.full_like(price, np.nan)
        kama_val[period] = np.mean(price[:period+1])
        for i in range(period+1, len(price)):
            kama_val[i] = kama_val[i-1] + sc[i] * (price[i] - kama_val[i-1])
        return kama_val
    
    # RSI (Relative Strength Index)
    def rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(price, np.nan)
        avg_loss = np.full_like(price, np.nan)
        if len(price) >= period + 1:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            for i in range(period+1, len(price)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    # Choppiness Index
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(
                high[i] - low[i],
                np.abs(high[i] - close[i-1]),
                np.abs(low[i] - close[i-1])
            )
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        # Chop calculation
        chop = np.full_like(close, 50.0)
        for i in range(period-1, len(close)):
            if hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop
    
    # Calculate 1d indicators
    kama_val = kama(close, period=10, fast=2, slow=30)
    rsi_val = rsi(close, period=14)
    chop_val = choppiness_index(high, low, close, period=14)
    
    # KAMA slope (direction)
    kama_slope = np.zeros_like(kama_val)
    kama_slope[1:] = np.diff(kama_val)
    
    # --- Weekly Trend Filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    # Weekly EMA20 for trend
    def ema(price, period):
        ema_val = np.full_like(price, np.nan)
        if len(price) >= period:
            ema_val[period-1] = np.mean(price[:period])
            alpha = 2 / (period + 1)
            for i in range(period, len(price)):
                ema_val[i] = alpha * price[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    weekly_ema20 = ema(weekly_close, 20)
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # --- Alignment ---
    kama_aligned = kama_val  # Already 1d
    rsi_aligned = rsi_val    # Already 1d
    chop_aligned = chop_val  # Already 1d
    kama_slope_aligned = kama_slope  # Already 1d
    
    # --- Signal Generation ---
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(weekly_ema20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Conditions
        kama_up = kama_slope_aligned[i] > 0
        kama_down = kama_slope_aligned[i] < 0
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        chop_high = chop_aligned[i] > 61.8  # Ranging market
        weekly_uptrend = close[i] > weekly_ema20_aligned[i]
        weekly_downtrend = close[i] < weekly_ema20_aligned[i]
        
        if position == 0:
            # Long: KAMA up, RSI > 50, choppy market, weekly uptrend
            if kama_up and rsi_above_50 and chop_high and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, choppy market, weekly downtrend
            elif kama_down and rsi_below_50 and chop_high and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down or RSI < 50 or weekly trend turns down
            if not kama_up or not rsi_above_50 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up or RSI > 50 or weekly trend turns up
            if not kama_down or not rsi_below_50 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals