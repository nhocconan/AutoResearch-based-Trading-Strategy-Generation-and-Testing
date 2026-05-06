#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop regime filter for trend following
# Uses Kaufman Adaptive Moving Average (KAMA) on 1d to identify trend direction
# Confirms with RSI(14) > 50 for long, < 50 for short
# Filters choppy markets using Choppiness Index (CI) > 61.8 = range (avoid), CI < 38.2 = trend (trade)
# Volume confirmation (>1.3x 20-bar average) ensures institutional participation
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in bull/bear: follows strong trends, avoids whipsaws in consolidation

name = "1d_KAMA_RSI_Chop_Volume_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on 1d timeframe
    def kama(data, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(data, period))
        volatility = np.sum(np.abs(np.diff(data)), axis=0) if len(data) > 1 else 0
        er = np.zeros_like(data)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama_val = np.full_like(data, np.nan)
        kama_val[period] = data[period]
        for i in range(period+1, len(data)):
            kama_val[i] = kama_val[i-1] + sc[i] * (data[i] - kama_val[i-1])
        return kama_val
    
    kama_1d = kama(close_1d, 10, 2, 30)
    
    # Calculate RSI(14) on 1d
    def rsi(data, period=14):
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = np.zeros_like(data)
        rsi_val[period:] = 100 - (100 / (1 + rs[period:]))
        return rsi_val
    
    rsi_1d = rsi(close_1d, 14)
    
    # Calculate Choppiness Index on 1w
    def choppiness_index(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of true ranges over period
        atr_sum = np.convolve(tr, np.ones(period), 'same')
        atr_sum[:period-1] = np.nan
        atr_sum[-period+1:] = np.nan
        
        # Highest high and lowest low over period
        hh = np.zeros_like(high)
        ll = np.zeros_like(low)
        for i in range(len(high)):
            if i >= period-1:
                hh[i] = np.max(high[i-period+1:i+1])
                ll[i] = np.min(low[i-period+1:i+1])
            else:
                hh[i] = np.nan
                ll[i] = np.nan
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if not np.isnan(tr[i]) and hh[i] > ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = np.nan
        return chop
    
    chop_1w = choppiness_index(high_1w, low_1w, close_1w, 14)
    
    # Calculate ATR(14) for 1d timeframe (for stoploss)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation filter (>1.3x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Align HTF indicators to 1d timeframe (primary)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w, additional_delay_bars=0)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above KAMA AND RSI > 50 AND trending market (CHOP < 38.2) AND volume confirmation
            if (close[i] > kama_1d_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                chop_1w_aligned[i] < 38.2 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA AND RSI < 50 AND trending market (CHOP < 38.2) AND volume confirmation
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  chop_1w_aligned[i] < 38.2 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below KAMA OR RSI < 45 OR choppy market (CHOP > 61.8)
            if (close[i] < kama_1d_aligned[i] or 
                rsi_1d_aligned[i] < 45 or 
                chop_1w_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above KAMA OR RSI > 55 OR choppy market (CHOP > 61.8)
            if (close[i] > kama_1d_aligned[i] or 
                rsi_1d_aligned[i] > 55 or 
                chop_1w_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals