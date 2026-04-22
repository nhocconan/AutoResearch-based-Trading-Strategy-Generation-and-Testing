#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for long-term trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_weekly = df_weekly['close'].values
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Load daily data for KAMA and RSI - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily KAMA(14) - Kaufman Adaptive Moving Average
    close_daily = df_daily['close'].values
    change = np.abs(np.diff(close_daily, prepend=close_daily[0]))
    volatility = np.abs(np.diff(close_daily))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2))**2
    kama = np.zeros_like(close_daily)
    kama[0] = close_daily[0]
    for i in range(1, len(close_daily)):
        kama[i] = kama[i-1] + sc[i] * (close_daily[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    
    # Calculate daily RSI(14)
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_50_weekly_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA50 + KAMA turning up + RSI > 50
            if (close[i] > ema_50_weekly_aligned[i] and 
                kama[i] > kama[i-1] and 
                rsi[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA50 + KAMA turning down + RSI < 50
            elif (close[i] < ema_50_weekly_aligned[i] and 
                  kama[i] < kama[i-1] and 
                  rsi[i] < 50):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Reverse conditions
            if position == 1:
                if (close[i] < ema_50_weekly_aligned[i] or 
                    kama[i] < kama[i-1] or 
                    rsi[i] < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > ema_50_weekly_aligned[i] or 
                    kama[i] > kama[i-1] or 
                    rsi[i] > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_KAMA_WeeklyTrend_RSI_Filter"
timeframe = "1d"
leverage = 1.0