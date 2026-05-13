#!/usr/bin/env python3
name = "12h_KAMA_Trend_1dRSI_Filter"
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
    
    # Load 1D data ONCE for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1D
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        # First values
        if len(gain) >= period:
            avg_gain[period-1] = np.nanmean(gain[1:period])
            avg_loss[period-1] = np.nanmean(loss[1:period])
            
            # Wilder's smoothing
            for i in range(period, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.full_like(close, np.nan)
        valid = ~np.isnan(avg_loss) & (avg_loss != 0)
        rs[valid] = avg_gain[valid] / avg_loss[valid]
        
        rsi = np.full_like(close, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Align 1D RSI to 12H timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 12H KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, er_period=10, fast_ema=2, slow_ema=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > er_period else np.array([])
        # Better approach: compute volatility as sum of absolute changes over er_period window
        volatility = np.array([np.sum(np.abs(np.diff(close[i-er_period+1:i+1])) if i >= er_period-1 else np.nan) for i in range(len(close))])
        er = np.where(volatility != 0, change / volatility, 0)
        er = np.concatenate([[np.nan] * (er_period-1), er]) if len(er) < len(close) else er
        
        # Smoothing constants
        sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
        
        kama = np.full_like(close, np.nan)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
        return kama
    
    kama_12h = calculate_kama(close, er_period=10, fast_ema=2, slow_ema=30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(kama_12h[i]):
            signals[i] = 0.0
            continue
        
        # RSI filter: Avoid extreme overbought/oversold
        rsi_ok = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama_12h[i]
        price_below_kama = close[i] < kama_12h[i]
        
        if position == 0:
            # LONG: Price above KAMA + not overbought
            if price_above_kama and rsi_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + not oversold
            elif price_below_kama and rsi_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or RSI overbought
            if price_below_kama or rsi_1d_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or RSI oversold
            if price_above_kama or rsi_1d_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals