#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI2 + 1d EMA200 filter for mean reversion in trending markets.
# Long when RSI2 < 10 and price > EMA200 (uptrend pullback).
# Short when RSI2 > 90 and price < EMA200 (downtrend bounce).
# Uses 1h for entry timing, 4h for RSI2 calculation, 1d for trend filter.
# Target: 15-35 trades/year per symbol with strict entry conditions.
name = "1h_RSI2_EMA200_MeanReversion"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for RSI2 calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate RSI(2) on 4h close
    def calculate_rsi(close, period=2):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_2_4h = calculate_rsi(close_4h, 2)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align indicators to 1h timeframe
    rsi_2_aligned = align_htf_to_ltf(prices, df_4h, rsi_2_4h)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure EMA200 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_2_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_2 = rsi_2_aligned[i]
        ema_200 = ema_200_aligned[i]
        
        if position == 0:
            # Enter long: RSI2 < 10 (oversold) and price above EMA200 (uptrend)
            if rsi_2 < 10 and price > ema_200:
                signals[i] = 0.20
                position = 1
            # Enter short: RSI2 > 90 (overbought) and price below EMA200 (downtrend)
            elif rsi_2 > 90 and price < ema_200:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when RSI2 > 50 (mean reversion complete) or price < EMA200
            if rsi_2 > 50 or price < ema_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when RSI2 < 50 (mean reversion complete) or price > EMA200
            if rsi_2 < 50 or price > ema_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals