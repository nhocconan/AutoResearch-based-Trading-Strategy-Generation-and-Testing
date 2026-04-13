#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop regime filter
# KAMA adapts to market efficiency - slow in ranging, fast in trending markets.
# RSI(14) identifies overbought/oversold conditions.
# Chop > 61.8 indicates ranging market (mean revert), Chop < 38.2 indicates trending (follow trend).
# Weekly trend filter avoids counter-trend trades.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA ( Kaufman Adaptive Moving Average )
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close, prepend=close[0]))
        for i in range(1, len(close)):
            volatility[i] += volatility[i-1]
        
        er = np.zeros_like(close)
        for i in range(er_length, len(close)):
            if volatility[i-er_length] != 0:
                er[i] = change[i] / volatility[i-er_length]
            else:
                er[i] = 0
        
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    # RSI
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        for i in range(1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    # Choppiness Index
    def chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        highest_high[0] = high[0]
        lowest_low[0] = low[0]
        for i in range(1, len(close)):
            highest_high[i] = max(highest_high[i-1], high[i])
            lowest_low[i] = min(lowest_low[i-1], low[i])
        
        chop_val = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if atr[i] > 0:
                chop_val[i] = 100 * np.log10(highest_high[i] - lowest_low[i]) / (np.log10(length) * np.log10(atr[i]))
            else:
                chop_val[i] = 50
        return chop_val
    
    # Calculate indicators
    kama_val = kama(close, 10, 2, 30)
    rsi_val = rsi(close, 14)
    chop_val = chop(high, low, close, 14)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(20) for trend filter
    ema20_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (20 + 1)
    ema20_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema20_1w[i] = (close_1w[i] - ema20_1w[i-1]) * ema_multiplier + ema20_1w[i-1]
    
    # Align weekly EMA to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(chop_val[i]) or np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_now = kama_val[i]
        rsi_now = rsi_val[i]
        chop_now = chop_val[i]
        weekly_trend = ema20_1w_aligned[i]
        
        if position == 0:
            # Long: Price > KAMA + RSI < 30 (oversold) + Chop > 61.8 (ranging) + Above weekly EMA
            if (price > kama_now and
                rsi_now < 30 and
                chop_now > 61.8 and
                price > weekly_trend):
                position = 1
                signals[i] = position_size
            # Short: Price < KAMA + RSI > 70 (overbought) + Chop > 61.8 (ranging) + Below weekly EMA
            elif (price < kama_now and
                  rsi_now > 70 and
                  chop_now > 61.8 and
                  price < weekly_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price < KAMA or RSI > 70 or Chop < 38.2 (trending) or Below weekly EMA
            if (price < kama_now or
                rsi_now > 70 or
                chop_now < 38.2 or
                price < weekly_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price > KAMA or RSI < 30 or Chop < 38.2 (trending) or Above weekly EMA
            if (price > kama_now or
                rsi_now < 30 or
                chop_now < 38.2 or
                price > weekly_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0