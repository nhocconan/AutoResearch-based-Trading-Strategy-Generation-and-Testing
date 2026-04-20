#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate weekly trend using 20-period EMA
    weekly_close = df_weekly['close'].values
    ema_20_weekly = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend = ema_20_weekly[-1] > ema_20_weekly[-2]  # upward if today > yesterday
    weekly_trend_array = np.full(len(weekly_close), weekly_trend, dtype=bool)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_array.astype(float))
    
    # Calculate daily ATR for volatility filter
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Wilder's smoothing for ATR
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_d = wilder_smooth(tr, 14)
    atr_d_aligned = align_htf_to_ltf(prices, df_daily, atr_d)
    
    # Calculate 6-period RSI for momentum
    close_p = prices['close'].values
    delta = np.diff(close_p, prepend=close_p[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average volume
    volume_p = prices['volume'].values
    vol_avg = pd.Series(volume_p).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close_p[i]
        rsi_val = rsi[i]
        weekly_trend_val = weekly_trend_aligned[i]
        atr_val = atr_d_aligned[i]
        vol_val = volume_p[i]
        vol_avg_val = vol_avg[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(weekly_trend_val) or 
            np.isnan(atr_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly uptrend, RSI > 50 (bullish momentum), volume above average
            if weekly_trend_val > 0.5 and rsi_val > 50 and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend, RSI < 50 (bearish momentum), volume above average
            elif weekly_trend_val < 0.5 and rsi_val < 50 and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Weekly trend turns down OR RSI < 40 (momentum loss)
            if weekly_trend_val < 0.5 or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Weekly trend turns up OR RSI > 60 (momentum loss)
            if weekly_trend_val > 0.5 or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyTrend_RSI_Momentum_Volume"
timeframe = "6h"
leverage = 1.0