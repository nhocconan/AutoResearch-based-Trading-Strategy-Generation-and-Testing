#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Relative Strength Index (RSI) with 12-hour trend filter.
# Long when RSI < 30 (oversold) and 12h price > 20-period EMA (uptrend).
# Short when RSI > 70 (overbought) and 12h price < 20-period EMA (downtrend).
# Exit when RSI returns to neutral zone (40-60).
# Uses RSI for mean reversion in ranging markets, EMA for trend filter to avoid counter-trend trades.
# Target: 12-37 trades/year to avoid fee drag. Works in bull/bear via trend-filtered mean reversion.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h EMA(20) for trend filter
    ema_period = 20
    ema = np.full(n, np.nan)
    if n >= ema_period:
        ema[ema_period-1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema[i] = (close[i] * 2 + ema[i-1] * (ema_period-1)) / (ema_period+1)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI
    def calculate_rsi(data, period=14):
        if len(data) < period + 1:
            return np.full(len(data), np.nan)
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(data), np.nan)
        avg_loss = np.full(len(data), np.nan)
        
        # First average is simple mean
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Subsequent values: Wilder's smoothing
        for i in range(period+1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI and EMA
    start_idx = max(14, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi = rsi_aligned[i]
        ema_val = ema[i]
        
        if position == 0:
            # Long: RSI oversold (<30) and price above EMA (uptrend)
            if rsi < 30 and price > ema_val:
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70) and price below EMA (downtrend)
            elif rsi > 70 and price < ema_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (>40) or trend breaks
            if rsi > 40 or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (<60) or trend breaks
            if rsi < 60 or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_RSI_OversoldOverbought_EMAFilter"
timeframe = "12h"
leverage = 1.0