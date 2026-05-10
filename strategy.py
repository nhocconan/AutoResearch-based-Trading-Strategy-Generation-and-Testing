#!/usr/bin/env python3
# 1d_1w_KAMA_RSI_ChopFilter
# Hypothesis: Daily KAMA trend direction filtered by weekly EMA for trend bias, with RSI mean reversion and Choppiness index regime filter to avoid whipsaws.
# Designed for low trade frequency (~15-25/year) to minimize fee drag in both bull and bear markets.
# Uses KAMA for adaptive trend, RSI(14) for mean reversion entries, and Choppiness index (>61.8) to identify ranging markets where mean reversion works best.

name = "1d_1w_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter and daily data for indicators
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA34 for trend bias (more stable than EMA50 for weekly)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # KAMA components: Efficiency Ratio and Smoothing Constants
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # will be corrected below
    # Correct volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    # For first 10 periods, use expanding sum
    for i in range(1, 10):
        volatility[i] = np.sum(np.abs(np.diff(close[:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for mean reversion
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close)
    rs[13:] = avg_gain[13:] / np.where(avg_loss[13:] == 0, 1e-10, avg_loss[13:])
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period) for regime detection
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    
    # ATR(14)
    atr = np.zeros_like(close)
    atr[13] = np.mean(tr[1:14])
    for i in range(14, len(close)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of True Range over 14 periods
    sum_tr = np.zeros_like(close)
    for i in range(13, len(close)):
        sum_tr[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(13, len(close)):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        if sum_tr[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(sum_tr[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when undefined
    
    # Align weekly EMA34 to daily
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from weekly EMA34
        bullish_trend = close[i] > ema_34_1w_aligned[i]
        bearish_trend = close[i] < ema_34_1w_aligned[i]
        
        # Mean reversion conditions: RSI extremes in choppy market
        # Choppiness > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop[i] > 61.8
        
        if position == 0:
            # Long: RSI oversold in bullish trend bias + ranging market
            if rsi[i] < 30 and bullish_trend and ranging_market:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in bearish trend bias + ranging market
            elif rsi[i] > 70 and bearish_trend and ranging_market:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: RSI reverts to midline or trend breaks
                if rsi[i] > 50 or not bullish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI reverts to midline or trend breaks
                if rsi[i] < 50 or not bearish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals