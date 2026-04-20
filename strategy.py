#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for primary trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for entries and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR calculation using Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_14 = wilder_smooth(tr, 14)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily Bollinger Bands (20, 2) for mean reversion signals
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Calculate daily RSI(14) for momentum confirmation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rsi_wilder(data, period):
        result = np.zeros_like(data)
        avg_gain = np.mean(data[:period])
        avg_loss = np.mean(data[:period])
        result[period-1] = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100
        for i in range(period, len(data)):
            avg_gain = (data[i] + (avg_gain * (period - 1))) / period
            avg_loss = (data[i] + (avg_loss * (period - 1))) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            result[i] = 100 - (100 / (1 + rs))
        return result
    
    rsi_14 = rsi_wilder(gain, 14)  # Actually calculates RSI on gains, need to fix
    # Recalculate properly
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (gain[i] + (avg_gain[i-1] * 13)) / 14
        avg_loss[i] = (loss[i] + (avg_loss[i-1] * 13)) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
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
        ema_50_val = ema_50_1w_aligned[i]
        atr_val = atr_14_aligned[i]
        upper_val = upper_band_aligned[i]
        lower_val = lower_band_aligned[i]
        rsi_val = rsi_14_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_50_val) or np.isnan(atr_val) or 
            np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Above weekly EMA50 (bullish trend), RSI < 30 (oversold), price touches lower BB
            if close_val > ema_50_val and rsi_val < 30 and close_val <= lower_val:
                signals[i] = 0.25
                position = 1
            # Short: Below weekly EMA50 (bearish trend), RSI > 70 (overbought), price touches upper BB
            elif close_val < ema_50_val and rsi_val > 70 and close_val >= upper_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 70 (overbought) or price touches upper BB
            if rsi_val > 70 or close_val >= upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 30 (oversold) or price touches lower BB
            if rsi_val < 30 or close_val <= lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_WeeklyEMA50_BB_RSI_MeanReversion_v1
# Uses weekly EMA(50) for trend filter (long above, short below)
# Uses daily Bollinger Bands (20,2) for mean reversion entries
# Uses daily RSI(14) for overbought/oversold confirmation
# Session filter: 8-20 UTC to avoid low-volume periods
# Designed for 12h timeframe with ~15-25 trades/year
name = "12h_WeeklyEMA50_BB_RSI_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0