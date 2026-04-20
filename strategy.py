#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Daily ATR-based volatility filter: current ATR > 1.5 * 20-day ATR average
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    vol_filter = atr_14_aligned > (1.5 * atr_ma_20_aligned)
    
    # Daily price position relative to weekly EMA
    price_above_ema = close_1d > ema_20_1w_aligned
    
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
        
        # Skip if any value is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly EMA, RSI < 30 (oversold), high volatility
            if price_above_ema[i] and rsi_14_aligned[i] < 30 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA, RSI > 70 (overbought), high volatility
            elif not price_above_ema[i] and rsi_14_aligned[i] > 70 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 70 (overbought) or price crosses below weekly EMA
            if rsi_14_aligned[i] > 70 or not price_above_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 30 (oversold) or price crosses above weekly EMA
            if rsi_14_aligned[i] < 30 or price_above_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_WeeklyEMA_RSI_VolatilityFilter_v1
# Uses weekly EMA(20) for trend filter
# Uses daily RSI(14) for overbought/oversold signals
# Requires volatility filter: current ATR > 1.5 * 20-day ATR average
# Session filter: 8-20 UTC to avoid low-volume periods
# Designed for 1d timeframe with ~10-25 trades/year
name = "1d_WeeklyEMA_RSI_VolatilityFilter_v1"
timeframe = "1d"
leverage = 1.0