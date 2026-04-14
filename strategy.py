#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR for volatility filter (14-period)
    tr_1w = np.zeros(len(df_1w))
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr_1w[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    atr_1d = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(df_1w)):
            ema50_1w[i] = (close_1w[i] + ema50_1w[i-1]) / 2
    
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily ATR for position sizing
    tr_1d = np.zeros(n)
    tr_1d[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1d[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr_daily = np.full(n, np.nan)
    if n >= 14:
        atr_daily[13] = np.mean(tr_1d[:14])
        for i in range(14, n):
            atr_daily[i] = (atr_daily[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate daily EMA200 for trend filter
    ema200_daily = np.full(n, np.nan)
    if n >= 200:
        ema200_daily[199] = np.mean(close[:200])
        for i in range(200, n):
            ema200_daily[i] = (close[i] + ema200_daily[i-1] * 199) / 200
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(200, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d[i]) or
            np.isnan(ema50_1d[i]) or
            np.isnan(atr_daily[i]) or
            np.isnan(ema200_daily[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_daily[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Dynamic position sizing based on volatility (inverse volatility)
        vol_factor = np.clip(0.01 * close[i] / atr_daily[i], 0.5, 2.0)
        position_size = base_size * vol_factor
        
        if position == 0:
            # Long: Price above weekly EMA50 AND above daily EMA200 with volatility filter
            if close[i] > ema50_1d[i] and close[i] > ema200_daily[i]:
                position = 1
                signals[i] = position_size
            # Short: Price below weekly EMA50 AND below daily EMA200 with volatility filter
            elif close[i] < ema50_1d[i] and close[i] < ema200_daily[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls below weekly EMA50 OR below daily EMA200
            if close[i] < ema50_1d[i] or close[i] < ema200_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises above weekly EMA50 OR above daily EMA200
            if close[i] > ema50_1d[i] or close[i] > ema200_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_EMA50_EMA200_Volatility_Scaled"
timeframe = "1d"
leverage = 1.0