#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12-hour EMA for trend direction and 1-day RSI for overbought/oversold conditions.
# Long when price is above 12h EMA and 1d RSI < 30 (oversold bounce in uptrend).
# Short when price is below 12h EMA and 1d RSI > 70 (overbought pullback in downtrend).
# Exit when price crosses the 12h EMA or RSI returns to neutral (40-60 range).
# Uses mean-reversion within trend to capture swings in both bull and bear markets.
# Designed for low trade frequency (<30/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate 1d RSI (14-period)
    rsi_period = 14
    rsi_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= rsi_period + 1:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(close_1d))
        avg_loss = np.zeros(len(close_1d))
        
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss[rsi_period:] != 0, 
                      avg_gain[rsi_period:] / avg_loss[rsi_period:], 0)
        rsi_1d[rsi_period:] = 100 - (100 / (1 + rs))
    
    # Align 12h EMA and 1d RSI to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA50 and RSI14
    start_idx = max(ema_period - 1, rsi_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema = ema_12h_aligned[i]
        rsi = rsi_1d_aligned[i]
        
        if position == 0:
            # Long: price above 12h EMA and 1d RSI oversold (<30)
            if price > ema and rsi < 30:
                signals[i] = size
                position = 1
            # Short: price below 12h EMA and 1d RSI overbought (>70)
            elif price < ema and rsi > 70:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 12h EMA or RSI returns to neutral (>=40)
            if price < ema or rsi >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 12h EMA or RSI returns to neutral (<=60)
            if price > ema or rsi <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_12hEMA50_1dRSI_MeanReversion"
timeframe = "4h"
leverage = 1.0