#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: Trade 4h KAMA trend with RSI mean reversion in choppy markets.
- Trend filter: 4h price > KAMA(10) = bullish, price < KAMA(10) = bearish.
- In bullish trend: buy when RSI(14) < 30 (oversold pullback).
- In bearish trend: sell when RSI(14) > 70 (overbought bounce).
- Chop filter: avoid trading when Choppiness Index(14) > 61.8 (strong ranging).
- Exit on trend reversal or RSI returning to neutral (40-60 range).
- Position size: 0.25. Target: 75-200 total trades over 4 years = 19-50/year.
- Works in both bull and bear: KAMA adapts to trend, RSI captures mean reversion in trends, chop filter avoids whipsaws in ranges.
"""

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
    
    # Calculate 4h KAMA(10) for trend
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.nansum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of abs changes
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants: fastest EMA(2), slowest EMA(30)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after first ER can be calculated
    for i in range(10, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 4h RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.nanmean(gain[1:15])  # First average
    avg_loss[13] = np.nanmean(loss[1:15])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h Choppiness Index(14)
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr1[0] = tr2[0] = tr3[0] = 0  # First period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) = sum of TR over 14 periods
    atr_14 = np.convolve(tr, np.ones(14)/14, mode='same')
    # For edges, use simple mean
    for i in range(14):
        atr_14[i] = np.nan
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh_14 = np.convolve(high, np.ones(14)/14, mode='same')
    ll_14 = np.convolve(low, np.ones(14)/14, mode='same')
    for i in range(14):
        hh_14[i] = ll_14[i] = np.nan
    for i in range(14, n):
        hh_14[i] = np.max(high[i-13:i+1])
        ll_14[i] = np.min(low[i-13:i+1])
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = np.convolve(tr, np.ones(14)/14, mode='same') * 14
    for i in range(14):
        sum_tr_14[i] = np.nan
    for i in range(14, n):
        sum_tr_14[i] = np.sum(tr[i-13:i+1])
    
    denominator = hh_14 - ll_14
    chop = np.where(denominator > 0, 
                    100 * np.log10(sum_tr_14 / denominator) / np.log10(14), 
                    50)  # Default to 50 when no range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA(10), RSI(14), Chop(14)
    start_idx = max(14, 10)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend and chop regime
        bullish_trend = close[i] > kama[i]
        bearish_trend = close[i] < kama[i]
        choppy_market = chop[i] > 61.8  # Strong ranging
        
        if position == 0:
            # Avoid trading in choppy markets
            if choppy_market:
                signals[i] = 0.0
                continue
                
            # Mean reversion entries in trending markets
            long_setup = bullish_trend and (rsi[i] < 30)
            short_setup = bearish_trend and (rsi[i] > 70)
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal or RSI returning to neutral
            exit_signal = (not bullish_trend) or (rsi[i] > 40 and rsi[i] < 60)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or RSI returning to neutral
            exit_signal = (not bearish_trend) or (rsi[i] > 40 and rsi[i] < 60)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0