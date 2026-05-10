#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On 1d timeframe, trade KAMA trend direction confirmed by RSI and filtered by Choppiness Index regime. Uses weekly trend filter for multi-timeframe alignment. KAMA adapts to market noise, reducing whipsaws in choppy markets. RSI provides momentum confirmation, while Choppiness Index avoids trend-following in ranging markets. Designed for low trade frequency (<25/year) to minimize fee drag, with discrete position sizing (0.25) to control drawdown. Works in both bull and bear markets via adaptive trend filter and regime detection.
"""

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for KAMA, RSI, and Choppiness Index
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - low)
    tr3 = np.abs(np.roll(low, 1) - high)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Avoid look-ahead: roll shifts forward, so we fix first element
    tr[0] = tr1[0]
    atr_sum = np.convolve(tr, np.ones(14), mode='full')[:n]  # Sum of TR over 14 periods
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    for i in range(n):
        start = max(0, i - 13)
        highest_high[i] = np.max(high[start:i+1])
        lowest_low[i] = np.min(low[start:i+1])
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    denominator = highest_high - lowest_low
    # Avoid division by zero
    chop = np.full_like(close, np.nan)
    mask = denominator > 0
    chop[mask] = 100 * np.log10(atrs_sum[mask] / denominator[mask]) / np.log10(14)
    
    # Align weekly EMA50 to daily
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), Chop (14), weekly EMA (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs weekly EMA50
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI conditions: avoid extremes, look for momentum
        rsi_bullish = 50 < rsi[i] < 70  # Not overbought, bullish momentum
        rsi_bearish = 30 < rsi[i] < 50  # Not oversold, bearish momentum
        
        # Choppiness filter: only trend-follow when trending (Chop < 38.2)
        trending_market = chop[i] < 38.2
        
        if position == 0:
            # Long: price above KAMA, uptrend, bullish RSI, trending market
            if price_above_kama and uptrend_1w and rsi_bullish and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, downtrend, bearish RSI, trending market
            elif price_below_kama and downtrend_1w and rsi_bearish and trending_market:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or trend fails or choppy market
            if price_below_kama or not uptrend_1w or chop[i] >= 61.8:  # Chop > 61.8 = ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or trend fails or choppy market
            if price_above_kama or not downtrend_1w or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals