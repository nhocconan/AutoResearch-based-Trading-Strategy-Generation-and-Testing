#!/usr/bin/env python3
name = "1d_KAMA_Trend_RSI_Pullback"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA: Kaufman Adaptive Moving Average
    def kama(arr, period=10, fast=2, slow=30):
        change = np.abs(np.diff(arr, n=period))
        volatility = np.sum(np.abs(np.diff(arr)), axis=1)
        er = np.zeros_like(arr)
        er[period:] = change / np.where(volatility[period:] == 0, 1, volatility[period:])
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(arr)
        kama[period] = arr[period]
        for i in range(period+1, len(arr)):
            kama[i] = kama[i-1] + sc[i] * (arr[i] - kama[i-1])
        return kama
    
    # RSI
    def rsi(arr, period=14):
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(arr)
        avg_loss = np.zeros_like(arr)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate 1-week EMA200 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate KAMA(10) and RSI(14) on daily
    kama_10 = kama(close, period=10, fast=2, slow=30)
    rsi_14 = rsi(close, period=14)
    
    # Volume filter: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(kama_10[i]) or 
            np.isnan(rsi_14[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        if position == 0:
            # Long: Price above KAMA + RSI oversold + above 1w EMA200 + volume
            if price_above_kama and rsi_oversold and close[i] > ema200_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + RSI overbought + below 1w EMA200 + volume
            elif price_below_kama and rsi_overbought and close[i] < ema200_1w_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below KAMA OR RSI overbought
                if close[i] < kama_10[i] or rsi_14[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: Price crosses above KAMA OR RSI oversold
                if close[i] > kama_10[i] or rsi_14[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals