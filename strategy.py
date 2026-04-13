#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volatility filter.
# Uses RSI(14) for momentum, 4h EMA(50) for trend, and 1d ATR ratio for volatility regime.
# In high volatility (ATR ratio > 1.2), we trade momentum pulls backs to EMA.
# In low volatility, we avoid trading to reduce whipsaw.
# Target: 15-30 trades per year (60-120 total over 4 years) for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # EMA(50) for 4h trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # ATR(14) for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # ATR ratio: current ATR / 50-period average ATR
    atr50 = pd.Series(atr14).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr14 / atr50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # RSI(14) for momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_trend = ema50_4h_aligned[i]
        vol_ratio = atr_ratio_aligned[i]
        
        # Only trade in high volatility regimes (avoid whipsaw in low vol)
        if vol_ratio < 1.2:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 40 (pullback) + price above 4h EMA50
            if rsi_val < 40 and price > ema_trend:
                position = 1
                signals[i] = position_size
            # Short: RSI > 60 (pullback) + price below 4h EMA50
            elif rsi_val > 60 and price < ema_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 60 (overbought) or price breaks below 4h EMA
            if rsi_val > 60 or price < ema_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 40 (oversold) or price breaks above 4h EMA
            if rsi_val < 40 or price > ema_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_RSI_EMA_ATR_Momentum"
timeframe = "1h"
leverage = 1.0