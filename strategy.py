#!/usr/bin/env python3
# 1d_KAMA_Trend_With_RSI_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# Combined with RSI for momentum confirmation and volatility filter to avoid whipsaws.
# Timeframe: 1d, uses weekly trend filter for multi-timeframe alignment.
# Low trade frequency (~10-20/year) via strict KAMA trend + RSI momentum + volatility filter.
# Long: KAMA rising, RSI > 50, and volatility below median (calm market).
# Short: KAMA falling, RSI < 50, and volatility below median.
# Exit: Opposite KAMA direction or volatility spike.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - adapts to market noise
    def calculate_kama(close, period=10, fast=2, slow=30):
        # Direction
        change = np.abs(np.diff(close, n=period))
        # Volatility
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        # Avoid division by zero
        volatility = np.where(volatility == 0, 1, volatility)
        # Efficiency Ratio
        er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
        # Smoothing Constants
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        # Initialize KAMA
        kama = np.full_like(close, np.nan, dtype=float)
        kama[period] = close[period]
        # Calculate KAMA
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate KAMA
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # RSI for momentum confirmation
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Volatility filter: ATR-based, use median to avoid whipsaws in high volatility
    def calculate_atr(high, low, close, period=14):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros_like(close)
        atr[period] = np.mean(tr[:period])
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i-1]) / period
        return atr
    
    atr = calculate_atr(high, low, close, period=14)
    # Use median ATR over long period for volatility filter
    vol_median = np.zeros_like(close)
    for i in range(len(close)):
        if i >= 50:
            vol_median[i] = np.median(atr[max(0, i-49):i+1])
        else:
            vol_median[i] = np.mean(atr[max(0, i):i+1]) if np.any(~np.isnan(atr[max(0, i):i+1])) else 0
    
    # Weekly trend filter for multi-timeframe alignment
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure we have enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_median[i]) or vol_median[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Volatility filter: only trade when volatility is below median (calm market)
        low_volatility = atr[i] < vol_median[i]
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, low volatility, and weekly uptrend
            if kama_rising and rsi[i] > 50 and low_volatility and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI < 50, low volatility, and weekly downtrend
            elif kama_falling and rsi[i] < 50 and low_volatility and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA falling or volatility spike
            if not kama_rising or not low_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA rising or volatility spike
            if not kama_falling or not low_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals