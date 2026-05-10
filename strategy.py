#!/usr/bin/env python3
# 4H_KAMA_RSI_Chop_Bounce
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction on 4h,
# combined with RSI(14) for mean reversion and Choppiness Index for regime filtering.
# Enters long when price crosses above KAMA in trending market (CHOP < 38.2) and RSI < 50,
# enters short when price crosses below KAMA in trending market (CHOP < 38.2) and RSI > 50.
# Uses 12h EMA(50) as higher timeframe trend filter to avoid counter-trend trades.
# Designed to work in both bull and bear markets by avoiding choppy regimes and
# aligning with higher timeframe trend. Targets 20-40 trades per year on 4h timeframe.

name = "4H_KAMA_RSI_Chop_Bounce"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for higher timeframe trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate KAMA on 4h
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        # Vectorized volatility sum
        er = np.zeros_like(close)
        for i in range(period, len(close)):
            if volatility != 0:
                er[i] = change[i] / volatility
            else:
                er[i] = 0
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[period] = close[period]
        for i in range(period+1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Calculate RSI(14)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Wilder smoothing
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Calculate Choppiness Index(14)
    def choppiness_index(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of ATR
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if atr_sum[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_vals = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        is_trending = chop_vals[i] < 38.2
        
        # Price position relative to KAMA
        price_above_kama = close[i] > kama_vals[i]
        price_below_kama = close[i] < kama_vals[i]
        
        # Higher timeframe trend filter
        price_above_12h_ema = close[i] > ema_50_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_50_12h_aligned[i]
        
        # RSI conditions
        rsi_below_50 = rsi_vals[i] < 50
        rsi_above_50 = rsi_vals[i] > 50
        
        # Entry conditions
        if position == 0:
            # Long: price crosses above KAMA in uptrend (aligned with 12h EMA), trending market, RSI < 50
            if (price_above_kama and 
                price_above_12h_ema and 
                is_trending and 
                rsi_below_50):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA in downtrend (aligned with 12h EMA), trending market, RSI > 50
            elif (price_below_kama and 
                  price_below_12h_ema and 
                  is_trending and 
                  rsi_above_50):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA OR chop becomes too high (choppy market) OR RSI > 70
            if (price_below_kama or 
                chop_vals[i] > 61.8 or 
                rsi_vals[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA OR chop becomes too high (choppy market) OR RSI < 30
            if (price_above_kama or 
                chop_vals[i] > 61.8 or 
                rsi_vals[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals