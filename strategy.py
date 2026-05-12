#!/usr/bin/env python3
name = "1d_KAMA_RSI_ChopFilter_v2"
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
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.diff(close), 'axis') else np.sum(np.abs(np.diff(close)))
    # Handle 1D case for volatility calculation
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+er_len]))) for i in range(len(change))])
    er = np.where(volatility != 0, change / volatility, 0)
    # Pad ER array to match close length
    er = np.concatenate([np.full(er_len, np.nan), er])
    
    # Calculate smoothing constant sc
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]  # Initialize
    for i in range(er_len + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI calculation
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan)
        avg_loss = np.full_like(close_prices, np.nan)
        
        # Initial average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Choppiness Index calculation
    def calculate_chop(high_prices, low_prices, close_prices, period=14):
        tr1 = high_prices - low_prices
        tr2 = np.abs(high_prices - np.roll(close_prices, 1))
        tr3 = np.abs(low_prices - np.roll(close_prices, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        atr = np.full_like(close_prices, np.nan)
        for i in range(period, len(close_prices)):
            if i == period:
                atr[i] = np.mean(tr[1:i+1])
            else:
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        highest_high = np.full_like(close_prices, np.nan)
        lowest_low = np.full_like(close_prices, np.nan)
        for i in range(len(close_prices)):
            if i >= period:
                highest_high[i] = np.max(high_prices[i-period+1:i+1])
                lowest_low[i] = np.min(low_prices[i-period+1:i+1])
        
        chop = np.full_like(close_prices, np.nan)
        for i in range(period, len(close_prices)):
            if highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(atr[i] * period / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # Neutral when no range
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Weekly trend filter (1w HTF)
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not ready
        if np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: KAMA bullish, RSI oversold recovery, low chop (trending market)
            if (close[i] > kama[i] and 
                rsi[i] > 30 and rsi[i] < 50 and  # RSI recovering from oversold
                chop[i] < 50):  # Trending market (chop < 50)
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA bearish, RSI overbought decline, low chop (trending market)
            elif (close[i] < kama[i] and 
                  rsi[i] < 70 and rsi[i] > 50 and  # RSI declining from overbought
                  chop[i] < 50):  # Trending market (chop < 50)
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when KAMA turns bearish or chop increases (rangy market)
            if (close[i] < kama[i] or chop[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when KAMA turns bullish or chop increases (rangy market)
            if (close[i] > kama[i] or chop[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals