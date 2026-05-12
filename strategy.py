# 12h KAMA + RSI + Chop Filter Strategy
# Hypothesis: KAMA adapts to market volatility, providing better trend signals than traditional moving averages.
# Combined with RSI momentum filter and Chop filter for regime detection, this should work in both bull and bear markets
# by avoiding whipsaws in ranging markets and capturing strong trends. Target: 12-37 trades/year.

#!/usr/bin/env python3
name = "12h_KAMA_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily data for Chop filter and weekly for trend context
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA on 12h data
    def kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / volatility[period-1:]
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama_vals = np.zeros_like(price)
        kama_vals[period] = price[period]
        for i in range(period+1, len(price)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (price[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, period=10, fast=2, slow=30)
    
    # Calculate RSI(14)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, period=14)
    
    # Calculate Chop index from daily data
    def chop(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.max(tr[:3]) if len(tr) >= 3 else np.max(tr) if len(tr) > 0 else 0], tr])
        
        # Sum of true ranges
        atr_sum = np.zeros_like(close)
        for i in range(len(close)):
            start = max(0, i - period + 1)
            atr_sum[i] = np.sum(tr[start:i+1])
        
        # Max/min close over period
        max_close = np.zeros_like(close)
        min_close = np.zeros_like(close)
        for i in range(len(close)):
            start = max(0, i - period + 1)
            max_close[i] = np.max(close[start:i+1])
            min_close[i] = np.min(close[start:i+1])
        
        # Chop calculation
        chop_vals = np.zeros_like(close)
        for i in range(len(close)):
            if atr_sum[i] > 0 and max_close[i] != min_close[i]:
                chop_vals[i] = 100 * np.log10(atr_sum[i] / (max_close[i] - min_close[i])) / np.log10(period)
            else:
                chop_vals[i] = 50  # neutral
        return chop_vals
    
    chop_vals = chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_vals)
    
    # Weekly EMA for trend filter
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, Chop < 61.8 (trending), volume filter, weekly trend up
            if (close[i] > kama_vals[i] and 
                rsi_vals[i] > 50 and 
                chop_aligned[i] < 61.8 and 
                vol_filter[i] and 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, Chop < 61.8 (trending), volume filter, weekly trend down
            elif (close[i] < kama_vals[i] and 
                  rsi_vals[i] < 50 and 
                  chop_aligned[i] < 61.8 and 
                  vol_filter[i] and 
                  close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI < 40
            if close[i] < kama_vals[i] or rsi_vals[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI > 60
            if close[i] > kama_vals[i] or rsi_vals[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals