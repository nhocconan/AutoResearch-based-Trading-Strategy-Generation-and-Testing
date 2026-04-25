#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v2
Hypothesis: Daily KAMA trend direction combined with RSI mean reversion and choppiness regime filter.
KAMA adapts to market noise - trends when efficient, ranges when choppy. Long when KAMA upward + RSI<40 in trending regime (CHOP<38.2) or RSI>60 in ranging regime (CHOP>61.8). Short when KAMA downward + RSI>60 in trending or RSI<40 in ranging.
Uses 1-week EMA50 as additional trend filter to avoid counter-trend trades. Volume confirmation reduces false signals.
Target: 7-25 trades/year (30-100 over 4 years).
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
    
    # 1-day data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # sum of absolute daily changes
    # Pad arrays for alignment
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.divide(change, volatility_sum, out=np.full_like(change, np.nan), where=volatility_sum!=0)
    # Smoothing constants: SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align daily KAMA to 1d timeframe (no alignment needed as we're on 1d)
    kama_aligned = kama  # Already on daily timeframe
    
    # RSI calculation on daily close
    delta = np.diff(close_1d)
    delta = np.concatenate([np.full(1, np.nan), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    # Using 14-period chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([np.full(1, np.nan), close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([np.full(1, np.nan), close_1d[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((max_high - min_low) > 0, chop, 50.0)  # Default to 50 when no range
    
    # 1-week data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA(10), RSI(14), CHOP(14), EMA50(1w), volume MA(20)
    start_idx = max(10, 14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        if position == 0:
            # Long conditions:
            # Trending regime: KAMA up + RSI < 40 (pullback)
            # Ranging regime: KAMA up + RSI > 60 (mean reversion from overbought)
            long_trending = (close[i] > kama_aligned[i]) and (rsi[i] < 40) and is_trending
            long_ranging = (close[i] > kama_aligned[i]) and (rsi[i] > 60) and is_ranging
            long_setup = (long_trending or long_ranging) and volume_spike[i]
            
            # Short conditions:
            # Trending regime: KAMA down + RSI > 60 (pullback)
            # Ranging regime: KAMA down + RSI < 40 (mean reversion from oversold)
            short_trending = (close[i] < kama_aligned[i]) and (rsi[i] > 60) and is_trending
            short_ranging = (close[i] < kama_aligned[i]) and (rsi[i] < 40) and is_ranging
            short_setup = (short_trending or short_ranging) and volume_spike[i]
            
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
            # Exit: KAMA turns down OR regime shifts strongly against trend
            if (close[i] < kama_aligned[i]) or (chop[i] > 70):  # Strong ranging regime exit
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: KAMA turns up OR regime shifts strongly against trend
            if (close[i] > kama_aligned[i]) or (chop[i] > 70):  # Strong ranging regime exit
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0