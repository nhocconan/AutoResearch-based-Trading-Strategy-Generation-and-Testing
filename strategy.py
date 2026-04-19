#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h trend alignment using 12h EMA21 for trend direction,
# 4h KAMA momentum for entry timing, and volume confirmation with choppiness filter.
# Targets 25-40 trades/year (100-160 total over 4 years) with strict entry conditions.
# KAMA adapts to market noise, reducing false signals in chop. Works in bull/bear by following higher timeframe trends.
name = "4h_12h_EMA21_KAMA_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA21 trend (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # KAMA calculation
    def calculate_kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[period:] = change[period:] / volatility[period:]
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # Choppiness Index (14-period)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr[period:] = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        chop = 100 * np.log10(atr * period / (max_high - min_low)) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, period=14)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 12h EMA21 AND above KAMA AND chop < 61.8 (trending) with volume
            if (close[i] > ema_21_12h_aligned[i] and 
                close[i] > kama[i] and 
                chop[i] < 61.8 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 12h EMA21 AND below KAMA AND chop < 61.8 (trending) with volume
            elif (close[i] < ema_21_12h_aligned[i] and 
                  close[i] < kama[i] and 
                  chop[i] < 61.8 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or chop > 61.8 (choppy)
            if close[i] < kama[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or chop > 61.8 (choppy)
            if close[i] > kama[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals