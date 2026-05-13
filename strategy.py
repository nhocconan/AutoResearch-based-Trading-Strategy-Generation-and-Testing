#!/usr/bin/env python3
name = "12h_KAMA_Trend_1dRSI_1dATR_Filter_v1"
timeframe = "12h"
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
    
    # Load 1D data ONCE for RSI and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 28:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
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
    
    # Calculate ATR(14) on 1D
    def calculate_atr(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    rsi_1d = calculate_rsi(close_1d, 14)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align 1D indicators to 12H timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12H KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, period))
        change = np.concatenate([[np.nan] * period, change])
        
        volatility = np.abs(np.diff(close))
        volatility = np.concatenate([[np.nan], volatility])
        volatility_sum = np.full_like(close, np.nan)
        for i in range(period, len(close)):
            volatility_sum[i] = np.nansum(volatility[i-period+1:i+1])
        
        er = np.full_like(close, np.nan)
        er = np.divide(change, volatility_sum, out=np.zeros_like(change), where=volatility_sum!=0)
        
        # Smoothing Constants
        sc = np.full_like(close, np.nan)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        kama = np.full_like(close, np.nan)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if np.isnan(sc[i]):
                kama[i] = kama[i-1]
            else:
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_12h = calculate_kama(close, 10, 2, 30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(kama_12h[i])):
            signals[i] = 0.0
            continue
        
        # KAMA slope (direction)
        kama_slope = kama_12h[i] - kama_12h[i-1] if i > 0 else 0
        
        # RSI filter: avoid extremes
        rsi_not_extreme = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        # ATR-based momentum filter: price change > 0.5 * ATR
        price_change = abs(close[i] - close[i-1]) if i > 0 else 0
        atr_threshold = 0.5 * atr_1d_aligned[i]
        sufficient_momentum = price_change > atr_threshold
        
        if position == 0:
            # LONG: KAMA up + RSI not overbought + sufficient momentum
            if kama_slope > 0 and rsi_not_extreme and sufficient_momentum:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down + RSI not oversold + sufficient momentum
            elif kama_slope < 0 and rsi_not_extreme and sufficient_momentum:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down OR RSI overbought
            if kama_slope < 0 or rsi_1d_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up OR RSI oversold
            if kama_slope > 0 or rsi_1d_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals