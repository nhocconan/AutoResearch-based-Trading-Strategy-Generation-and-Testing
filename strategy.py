#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA direction + RSI + chop filter
    # KAMA adapts to market noise, reducing whipsaw in choppy markets.
    # RSI identifies overbought/oversold conditions for mean reversion.
    # Chop filter identifies trending vs ranging markets to apply appropriate logic.
    # This combination aims to capture trends while avoiding false signals in low volatility.
    # Target: 7-25 trades/year on 1d timeframe to minimize fee drag.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for KAMA, RSI, and chop calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Calculate change and volatility
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 1 else np.zeros_like(change)
        # Avoid division by zero
        volatility = np.where(volatility == 0, 1, volatility)
        # Efficiency ratio
        er = change / volatility
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate RSI (Relative Strength Index)
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        # Use exponential moving average for average gain/loss
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        for i in range(1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        # Avoid division by zero
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Chopiness Index
    def calculate_chop(high, low, close, length=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Sum of True Range over period
        atr_sum = np.zeros_like(close)
        for i in range(length-1, len(close)):
            atr_sum[i] = np.sum(tr[i-length+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(length-1, len(close)):
            highest_high[i] = np.max(high[i-length+1:i+1])
            lowest_low[i] = np.min(low[i-length+1:i+1])
        
        # Chop calculation
        # Avoid division by zero
        hh_ll = highest_high - lowest_low
        hh_ll = np.where(hh_ll == 0, 1, hh_ll)
        chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(length)
        return chop
    
    # Calculate indicators on 1d data
    kama_1d = calculate_kama(close_1d, length=10, fast=2, slow=30)
    rsi_1d = calculate_rsi(close_1d, length=14)
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, length=14)
    
    # Align 1d indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Price above KAMA (uptrend), RSI not overbought, and trending market (low chop)
            if close[i] > kama_aligned[i] and rsi_aligned[i] < 70 and chop_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short conditions: Price below KAMA (downtrend), RSI not oversold, and trending market (low chop)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] > 30 and chop_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Trend reversal or overextended RSI
            if position == 1:
                if close[i] < kama_aligned[i] or rsi_aligned[i] > 80:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_aligned[i] or rsi_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0