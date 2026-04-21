#!/usr/bin/env python3
"""
6h_RSI_PriceAction_Confluence_V1
Hypothesis: Combine RSI extremes with price action patterns (inside bars and engulfing candles) on 6h timeframe, filtered by daily trend (EMA50). 
In bull markets: look for bullish engulfing at RSI < 30. In bear markets: look for bearish engulfing at RSI > 70.
Inside bars provide low-risk entry points with tight stops. Works in both regimes by adapting to higher timeframe trend.
Target: 15-30 trades/year per symbol with disciplined entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate RSI with proper Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing: alpha = 1/period
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # First average: simple mean
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    # Subsequent: Wilder's smoothing
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on daily close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros_like(close_1d)
    ema50_1d[:] = np.nan
    
    # Calculate EMA manually with proper initialization
    alpha = 2.0 / (50 + 1)
    ema50_1d[49] = np.mean(close_1d[:50])  # First EMA value
    for i in range(50, len(close_1d)):
        ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h data
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Calculate RSI(14) on 6h
    rsi_6h = calculate_rsi(close_6h, 14)
    
    # Price action patterns
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = np.zeros(n, dtype=bool)
    bearish_engulf = np.zeros(n, dtype=bool)
    inside_bar = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        # Bullish engulfing
        if (close_6h[i] > open_6h[i] and  # current bullish
            open_6h[i] <= close_6h[i-1] and  # current open <= prev close
            close_6h[i] >= open_6h[i-1] and  # current close >= prev open
            open_6h[i-1] > close_6h[i-1]):   # previous bearish
            bullish_engulf[i] = True
        
        # Bearish engulfing
        if (close_6h[i] < open_6h[i] and  # current bearish
            open_6h[i] >= close_6h[i-1] and  # current open >= prev close
            close_6h[i] <= open_6h[i-1] and  # current close <= prev open
            open_6h[i-1] < close_6h[i-1]):   # previous bullish
            bearish_engulf[i] = True
        
        # Inside bar: current range within previous bar's range
        if (high_6h[i] <= high_6h[i-1] and low_6h[i] >= low_6h[i-1]):
            inside_bar[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if EMA50 not available
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        rsi = rsi_6h[i]
        trend_up = price > ema50_1d_aligned[i]
        
        if position == 0:
            # Long conditions: bullish engulfing OR inside bar + RSI oversold in uptrend
            if (bullish_engulf[i] or inside_bar[i]) and rsi < 35 and trend_up:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish engulfing OR inside bar + RSI overbought in downtrend
            elif (bearish_engulf[i] or inside_bar[i]) and rsi > 65 and not trend_up:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought OR bearish engulfing
            if rsi > 70 or bearish_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold OR bullish engulfing
            if rsi < 30 or bullish_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_PriceAction_Confluence_V1"
timeframe = "6h"
leverage = 1.0