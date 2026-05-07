#!/usr/bin/env python3
# 6h_Keltner_Channel_RSI_Pullback_v1
# Hypothesis: Combines 6h Keltner Channel (20, 2) with RSI(14) pullback and 1d trend filter.
# Enters long when price touches lower KC band during uptrend (1d EMA50 up) with RSI<40,
# enters short when price touches upper KC band during downtrend (1d EMA50 down) with RSI>60.
# Uses volume confirmation to avoid false breakouts. Designed for 60-100 trades/year.

name = "6h_Keltner_Channel_RSI_Pullback_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Keltner Channel (20, 2) on 6h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(np.maximum(
        np.maximum(high - low, np.abs(high - np.roll(close, 1))),
        np.abs(low - np.roll(close, 1))
    )).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 2 * atr
    kc_lower = ema_20 - 2 * atr
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            vol_ok = volume[i] > vol_ma[i]
            
            # Long: price at lower KC band, uptrend (1d EMA50 rising), RSI oversold
            if (close[i] <= kc_lower[i] * 1.001 and  # allow small tolerance
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                rsi[i] < 40 and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price at upper KC band, downtrend (1d EMA50 falling), RSI overbought
            elif (close[i] >= kc_upper[i] * 0.999 and  # allow small tolerance
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                  rsi[i] > 60 and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses above EMA20 or RSI overbought
            if (close[i] >= ema_20[i] or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses below EMA20 or RSI oversold
            if (close[i] <= ema_20[i] or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals