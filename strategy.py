#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_And_Volume_Spice
Hypothesis: 4h KAMA trend direction + RSI(14) pullback + volume spike confirmation.
Uses KAMA for adaptive trend filtering, enters on RSI pullbacks with volume confirmation.
Designed for fewer trades (15-30/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend filter
    def kama(price, period=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / (volatility[period-1:] + 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(price)
        kama[period] = price[period]
        for i in range(period+1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama_values = kama(close, period=10, fast=2, slow=30)
    
    # RSI(14)
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
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_values = rsi(close, period=14)
    
    # Volume spike confirmation (2.0x 20-period average)
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma_20[:10] = np.nan
    vol_ma_20[-10:] = np.nan
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_values[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # KAMA trend direction
        trend_up = close[i] > kama_values[i]
        trend_down = close[i] < kama_values[i]
        
        # RSI conditions for pullback entries
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        # Entry logic: trend + RSI pullback + volume spike
        long_entry = trend_up and rsi_oversold and vol_spike[i]
        short_entry = trend_down and rsi_overbought and vol_spike[i]
        
        # Exit logic: opposite RSI extreme or trend reversal
        long_exit = rsi_values[i] > 70 or not trend_up
        short_exit = rsi_values[i] < 30 or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_RSI_And_Volume_Spice"
timeframe = "4h"
leverage = 1.0