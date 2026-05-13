#!/usr/bin/env python3
name = "1d_KAMA_Trend_1wATR_Filter"
timeframe = "1d"
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
    
    # Load 1W data ONCE for ATR filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough for ATR(14)
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(14) on 1W
    def calculate_atr(high, low, close, period=14):
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
    
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, 14)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate KAMA on 1D
    def kama(close, er_period=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_period))
        change = np.concatenate([[np.nan] * er_period, change])
        volatility = np.abs(np.diff(close))
        volatility = np.concatenate([[np.nan], volatility])
        
        er = np.full_like(close, np.nan)
        if len(volatility) >= er_period:
            vol_sum = np.nansum(volatility[1:er_period+1]) if er_period > 0 else 0
            if vol_sum > 0:
                er[er_period] = change[er_period] / vol_sum
            for i in range(er_period+1, len(close)):
                vol_sum = vol_sum - volatility[i-er_period] + volatility[i]
                if vol_sum > 0:
                    er[i] = change[i] / vol_sum
        
        # Smoothing Constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA
        kama_val = np.full_like(close, np.nan)
        kama_val[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(sc[i]):
                kama_val[i] = kama_val[i-1] + sc[i] * (close[i] - kama_val[i-1])
            else:
                kama_val[i] = kama_val[i-1]
        return kama_val
    
    kama_val = kama(close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data
        if (np.isnan(kama_val[i]) or np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to KAMA
        price_above_kama = close[i] > kama_val[i]
        price_below_kama = close[i] < kama_val[i]
        
        # ATR filter: only trade when volatility is elevated (above average)
        # Using 50-period average of ATR as threshold
        if i >= 50:
            atr_avg = np.nanmean(atr_1w_aligned[i-50:i])
            high_volatility = atr_1w_aligned[i] > atr_avg
        else:
            high_volatility = True  # Allow trading until we have enough data for average
        
        if position == 0:
            # LONG: price above KAMA + high volatility
            if price_above_kama and high_volatility:
                signals[i] = 0.25
                position = 1
            # SHORT: price below KAMA + high volatility
            elif price_below_kama and high_volatility:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA or volatility drops
            if not price_above_kama or not high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA or volatility drops
            if not price_below_kama or not high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals