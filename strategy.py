#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # Calculate weekly RSI (14-period) for trend filter
    delta = np.diff(df_1w['close'].values)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(df_1w), np.nan)
    avg_loss = np.full(len(df_1w), np.nan)
    
    if len(df_1w) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, len(df_1w)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14w = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 12h timeframe
    rsi_12w = align_htf_to_ltf(prices, df_1w, rsi_14w)
    
    # Calculate weekly ATR (14-period) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    atr_12w = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 12-hour Donchian channels (20-period) for entry signals
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):  # Start after sufficient warmup
        # Skip if any critical data is NaN
        if (np.isnan(rsi_12w[i]) or
            np.isnan(atr_12w[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (weekly ATR < 0.5% of price)
        if atr_12w[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of weekly RSI
        # RSI > 50 = bullish bias, RSI < 50 = bearish bias
        if position == 0:
            # Long: Price breaks above 12h Donchian high AND weekly RSI > 50
            if close[i] > donch_high[i] and rsi_12w[i] > 50:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 12h Donchian low AND weekly RSI < 50
            elif close[i] < donch_low[i] and rsi_12w[i] < 50:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 12h Donchian low OR weekly RSI turns bearish (< 40)
            if close[i] < donch_low[i] or rsi_12w[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 12h Donchian high OR weekly RSI turns bullish (> 60)
            if close[i] > donch_high[i] or rsi_12w[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Donchian_RSI_Trend"
timeframe = "12h"
leverage = 1.0