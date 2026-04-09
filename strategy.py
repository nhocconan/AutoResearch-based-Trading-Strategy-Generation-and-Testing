#!/usr/bin/env python3
# 6h_1d_rsi_divergence_v1
# Hypothesis: 6-hour RSI(14) divergence with price confirmed by 1-day trend filter and volume spike.
# Bullish divergence: price makes lower low, RSI makes higher low → long when price > 1-day EMA50 and volume > 1.5x avg.
# Bearish divergence: price makes higher high, RSI makes lower high → short when price < 1-day EMA50 and volume > 1.5x avg.
# Works in bull markets by catching bounces from oversold, in bear markets by selling rallies from overbought.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rsi_divergence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # Wilder's smoothing
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        rsi_vals[:period] = np.nan
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA50 on 1d
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[:] = np.nan
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / 51) + (ema_50_1d[i-1] * 49 / 51)
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(rsi_vals[i]) or np.isnan(rsi_vals[i-1]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 or price below 1d EMA50
            if rsi_vals[i] < 50 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 or price above 1d EMA50
            if rsi_vals[i] > 50 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Bullish divergence: price lower low, RSI higher low
            if (low[i] < low[i-1] and 
                rsi_vals[i] > rsi_vals[i-1] and
                rsi_vals[i] < 40 and  # Oversold threshold
                close[i] > ema_50_aligned[i] and  # Above 1d trend
                volume[i] > vol_ma_20[i] * 1.5):  # Volume confirmation
                position = 1
                signals[i] = 0.25
            # Bearish divergence: price higher high, RSI lower high
            elif (high[i] > high[i-1] and 
                  rsi_vals[i] < rsi_vals[i-1] and
                  rsi_vals[i] > 60 and  # Overbought threshold
                  close[i] < ema_50_aligned[i] and  # Below 1d trend
                  volume[i] > vol_ma_20[i] * 1.5):  # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals