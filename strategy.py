#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for entry logic
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly trend: 50-period EMA
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily ATR for volatility and position sizing
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14) using Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily RSI(14) for mean reversion signals
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rsi_wilder(data, period):
        result = np.zeros_like(data)
        avg_gain = np.mean(data[:period])
        avg_loss = np.mean(data[:period])
        result[period-1] = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 50
        for i in range(period, len(data)):
            avg_gain = (avg_gain * (period-1) + data[i]) / period
            avg_loss = (avg_loss * (period-1) + data[i]) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            result[i] = 100 - (100 / (1 + rs)) if rs != 0 else 100
        return result
    
    rsi_14 = rsi_wilder(gain, 14) - rsi_wilder(loss, 14) + 50  # Adjust to get proper RSI
    rsi_14 = np.where(rsi_14 < 0, 0, np.where(rsi_14 > 100, 100, rsi_14))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        weekly_trend = ema_50_1w_aligned[i]
        atr_val = atr_1d_aligned[i]
        rsi_val = rsi_14_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(weekly_trend) or np.isnan(atr_val) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Above weekly EMA50 (uptrend) and RSI oversold (<30)
            if close_val > weekly_trend and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: Below weekly EMA50 (downtrend) and RSI overbought (>70)
            elif close_val < weekly_trend and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or price drops below weekly EMA50
            if rsi_val > 70 or close_val < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or price rises above weekly EMA50
            if rsi_val < 30 or close_val > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_WeeklyEMA50_RSI14_MeanReversion_v1
# Uses weekly EMA50 for trend filter
# Uses daily RSI(14) for mean reversion entries
# Long: Price > weekly EMA50 AND RSI < 30 (oversold in uptrend)
# Short: Price < weekly EMA50 AND RSI > 70 (overbought in downtrend)
# Exits when RSI reverses or price crosses weekly EMA50
# Session filter: 8-20 UTC to avoid low-volume periods
# Designed for 1d timeframe with ~10-25 trades/year
name = "1d_WeeklyEMA50_RSI14_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0