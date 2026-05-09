#!/usr/bin/env python3
# 1D_KAMA_Trend_RSI_Extremes_ChopFilter_v2
# Hypothesis: Daily KAMA trend direction combined with RSI extremes and Choppiness regime filter.
# Works in bull/bear: KAMA adapts to trend strength, RSI extremes signal mean reversion opportunities,
# Choppiness filter avoids whipsaws in ranging markets. Uses 1h timeframe for entry timing.

name = "1D_KAMA_Trend_RSI_Extremes_ChopFilter_v2"
timeframe = "1h"
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
    
    # Calculate 1d KAMA for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    kama_1d = np.full_like(close_1d, np.nan)
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    if len(close_1d) >= er_period:
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close_1d, n=er_period))
        volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else 0
        er = np.zeros_like(close_1d)
        for i in range(er_period, len(close_1d)):
            price_change = np.abs(close_1d[i] - close_1d[i-er_period])
            price_volatility = np.sum(np.abs(np.diff(close_1d[i-er_period:i+1])))
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 0
        
        # Smoothing constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Initialize KAMA
        kama_1d[er_period] = close_1d[er_period]
        for i in range(er_period + 1, len(close_1d)):
            kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    
    # Align KAMA to 1h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1h RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    rsi_period = 14
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(close) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[0:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[0:rsi_period])
        for i in range(rsi_period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d Choppiness Index
    atr_period = 14
    high_low = np.maximum(high, np.roll(high, 1))
    high_low[0] = high[0]
    low_close = np.minimum(low, np.roll(close, 1))
    low_close[0] = low[0]
    tr = np.maximum(high_low - low_close, np.roll(high_low - low_close, 1))
    tr[0] = high[0] - low[0]
    
    atr = np.full_like(close, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[0:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate rolling max/min for Chop
    max_high = np.full_like(high, np.nan)
    min_low = np.full_like(low, np.nan)
    
    chop_period = 14
    if len(high) >= chop_period:
        for i in range(chop_period-1, len(high)):
            max_high[i] = np.max(high[i-chop_period+1:i+1])
            min_low[i] = np.min(low[i-chop_period+1:i+1])
    
    # Chop calculation
    sum_tr = np.full_like(close, np.nan)
    if len(tr) >= chop_period:
        sum_tr[chop_period-1] = np.sum(tr[0:chop_period])
        for i in range(chop_period, len(tr)):
            sum_tr[i] = sum_tr[i-1] - tr[i-chop_period] + tr[i]
    
    chop = np.full_like(close, 50.0)
    valid = (~np.isnan(sum_tr)) & (~np.isnan(max_high)) & (~np.isnan(min_low)) & ((max_high - min_low) > 0)
    chop[valid] = 100 * np.log10(sum_tr[valid] / (max_high[valid] - min_low[valid])) / np.log10(chop_period)
    
    # Align Chop to 1h timeframe
    chop_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, rsi_period, chop_period)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA (uptrend) AND RSI oversold AND chop > 61.8 (ranging)
            if (close[i] > kama_1d_aligned[i] and 
                rsi[i] < 30 and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) AND RSI overbought AND chop > 61.8 (ranging)
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi[i] > 70 and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought OR chop < 38.2 (trending) OR price crosses below KAMA
            if (rsi[i] > 70 or 
                chop_aligned[i] < 38.2 or 
                close[i] < kama_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold OR chop < 38.2 (trending) OR price crosses above KAMA
            if (rsi[i] < 30 or 
                chop_aligned[i] < 38.2 or 
                close[i] > kama_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals