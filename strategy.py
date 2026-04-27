#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    4h KAMA with RSI and Chop regime.
    Hypothesis: KAMA adapts to market regime (trend vs range), RSI filters extreme readings,
    Chop filter distinguishes trending from ranging markets. Works in both bull/bear by adapting.
    Long when: KAMA rising + RSI in neutral zone (40-60) + Chop > 61.8 (range) or Chop < 38.2 (trend)
    Short when: KAMA falling + RSI in neutral zone (40-60) + Chop > 61.8 (range) or Chop < 38.2 (trend)
    Exit: Opposite KAMA direction or Chop extreme reversal.
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on 4h close
    def kama(close, er_period=10, fast_ema=2, slow_ema=30):
        n = len(close)
        kama_arr = np.full(n, np.nan)
        if n < er_period:
            return kama_arr
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if n >= er_period else np.full(n-1, np.nan)
        # For each point, calculate ER
        er = np.full(n, np.nan)
        for i in range(er_period, n):
            if volatility[i-er_period:i] > 0:
                er[i] = change[i-er_period] / np.sum(np.abs(np.diff(close[i-er_period:i])))
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
        
        # Initialize KAMA
        kama_arr[er_period] = close[er_period]
        for i in range(er_period+1, n):
            if not np.isnan(sc[i]):
                kama_arr[i] = kama_arr[i-1] + sc[i] * (close[i] - kama_arr[i-1])
            else:
                kama_arr[i] = kama_arr[i-1]
        return kama_arr
    
    kama_val = kama(close, 10, 2, 30)
    
    # Calculate RSI(14)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        # Initial average
        if len(close) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_val = rsi(close, 14)
    
    # Calculate Chop on daily data
    def chop(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Sum of TR over period
        tr_sum = np.full(len(close), np.nan)
        for i in range(period, len(close)):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.full(len(close), np.nan)
        ll = np.full(len(close), np.nan)
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop_val = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
        return chop_val
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    chop_val = chop(high_1d, low_1d, close_1d, 14)
    
    # Align daily Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_val)
    
    # Calculate ATR for stop
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup
    start_idx = max(30, 14, 14) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction
        kama_rising = kama_val[i] > kama_val[i-1]
        kama_falling = kama_val[i] < kama_val[i-1]
        
        # RSI in neutral zone (avoid extremes)
        rsi_neutral = (rsi_val[i] >= 40) & (rsi_val[i] <= 60)
        
        # Chop regime: >61.8 = range, <38.2 = trend
        chop_range = chop_aligned[i] > 61.8
        chop_trend = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: KAMA rising + RSI neutral + (range OR trend)
            if kama_rising and rsi_neutral and (chop_range or chop_trend):
                signals[i] = size
                position = 1
            # Short: KAMA falling + RSI neutral + (range OR trend)
            elif kama_falling and rsi_neutral and (chop_range or chop_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: KAMA falling OR Chop extreme reversal
            if kama_falling or (chop_aligned[i] > 61.8 and chop_aligned[i-1] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: KAMA rising OR Chop extreme reversal
            if kama_rising or (chop_aligned[i] < 38.2 and chop_aligned[i-1] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_RSI_Chop_Regime"
timeframe = "4h"
leverage = 1.0