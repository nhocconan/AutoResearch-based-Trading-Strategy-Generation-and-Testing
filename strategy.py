#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_KAMA_Band_WithVolume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(close)
    er[er_period:] = change[er_period:] / (volatility[er_period:] + 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(close)
    atr_period = 14
    atr_sum = np.nansum(tr[:atr_period])
    if not np.isnan(atr_sum):
        atr[atr_period-1] = atr_sum / atr_period
        for i in range(atr_period, n):
            if not np.isnan(tr[i]):
                atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
            else:
                atr[i] = atr[i-1]
    
    # Weekly trend filter: EMA50 on weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current > 1.5 * 20-period average
    vol_ma20 = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period, 50)  # Ensure enough data for KAMA and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(kama[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions: Price above KAMA + Weekly uptrend + Volume spike
            if (price > kama[i] and 
                price > ema50_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short conditions: Price below KAMA + Weekly downtrend + Volume spike
            elif (price < kama[i] and 
                  price < ema50_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: Price below KAMA or weekly trend turns down
            if (price < kama[i] or 
                price < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price above KAMA or weekly trend turns up
            if (price > kama[i] or 
                price > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals