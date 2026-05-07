#!/usr/bin/env python3
name = "4h_RSI_Overbought_Oversold_With_Volume_Trend_Filter"
timeframe = "4h"
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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d RSI with mean reversion (oversold = buy, overbought = sell)
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(df_1d['close'].values, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d trend filter: price above/below 50 EMA for trend direction
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h volume confirmation: volume > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm_4h = volume > 1.5 * vol_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for RSI and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) in uptrend (price > EMA50) with volume confirmation
            if (rsi_1d_aligned[i] < 30 and close[i] > ema50_1d_aligned[i] and vol_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) in downtrend (price < EMA50) with volume confirmation
            elif (rsi_1d_aligned[i] > 70 and close[i] < ema50_1d_aligned[i] and vol_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI overbought (>70) or trend reversal (price < EMA50)
            if rsi_1d_aligned[i] > 70 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI oversold (<30) or trend reversal (price > EMA50)
            if rsi_1d_aligned[i] < 30 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI mean reversion works in both bull and bear markets when combined with trend filter.
# In bull markets: buy oversold dips in uptrend. In bear markets: sell overbought rallies in downtrend.
# Volume confirmation reduces false signals. Target: 20-40 trades/year to minimize fee drag.
# Position size 0.25 limits risk. Exit on mean reversion completion or trend change.