#!/usr/bin/env python3
# 1h_4d_rsi_divergence_mean_reversion_v1
# Hypothesis: Mean reversion on 1h using RSI divergence with 4h trend filter and daily volatility regime.
# Long when: RSI < 30 (oversold) + bullish divergence (price makes lower low, RSI makes higher low) + 4h close > 4h EMA50 (uptrend filter) + daily ATR ratio < 0.8 (low volatility regime).
# Short when: RSI > 70 (overbought) + bearish divergence (price makes higher high, RSI makes lower high) + 4h close < 4h EMA50 (downtrend filter) + daily ATR ratio < 0.8.
# Exit when RSI returns to 50 (mean reversion) or trend filter fails.
# Designed to generate ~20-40 trades/year to avoid fee drag while capturing reversals in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_rsi_divergence_mean_reversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate RSI (14) on 1h
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        rsi_vals[:period] = np.nan
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Calculate EMA50 on 4h for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = np.zeros_like(close_4h)
    ema_4h[0] = close_4h[0]
    alpha = 2 / (50 + 1)
    for i in range(1, len(close_4h)):
        ema_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_4h[i-1]
    ema_4h[:49] = np.nan
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate ATR and its ratio for volatility regime on daily
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR (14)
    atr_1d = np.zeros_like(tr)
    atr_1d[14] = np.mean(tr[1:15])
    for i in range(15, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    atr_1d[:14] = np.nan
    
    # ATR ratio: current ATR / 50-period average ATR
    atr_ma_50 = np.zeros_like(atr_1d)
    for i in range(50, len(atr_1d)):
        if not np.isnan(atr_1d[i-50:i]).any():
            atr_ma_50[i] = np.mean(atr_1d[i-50:i])
        else:
            atr_ma_50[i] = np.nan
    atr_ratio = atr_1d / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(rsi_vals[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or i < 14):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_vals[i]
        ema_4h_val = ema_4h_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        if position == 1:  # Long
            # Exit: RSI returns to 50 or trend filter fails
            if rsi_val >= 50 or close < ema_4h_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: RSI returns to 50 or trend filter fails
            if rsi_val <= 50 or close > ema_4h_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Check for divergences (need lookback)
            if i >= 2:
                # Bullish divergence: price lower low, RSI higher low
                bull_div = (low[i] < low[i-1] and low[i-1] < low[i-2] and 
                           rsi_val > rsi_vals[i-1] and rsi_vals[i-1] > rsi_vals[i-2])
                # Bearish divergence: price higher high, RSI lower high
                bear_div = (high[i] > high[i-1] and high[i-1] > high[i-2] and 
                           rsi_val < rsi_vals[i-1] and rsi_vals[i-1] < rsi_vals[i-2])
                
                # Entry conditions
                if (rsi_val < 30 and bull_div and 
                    close > ema_4h_val and atr_ratio_val < 0.8):
                    position = 1
                    signals[i] = 0.20
                elif (rsi_val > 70 and bear_div and 
                      close < ema_4h_val and atr_ratio_val < 0.8):
                    position = -1
                    signals[i] = -0.20
    
    return signals