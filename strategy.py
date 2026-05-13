#!/usr/bin/env python3
"""
1h_12h_4h_RSI_MeanReversion
Hypothesis: Mean reversion on 1h using RSI(14) filtered by 12h EMA200 trend and 4h volatility filter. 
In bull/bear markets, price tends to revert to mean during strong trends when overextended.
Uses 12h EMA200 for trend direction (long only in uptrend, short only in downtrend) and 
4h ATR-based volatility filter to avoid chop. Targets 15-35 trades/year with strict entry conditions.
"""

name = "1h_12h_4h_RSI_MeanReversion"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA200 for trend filter
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Get 4h data for volatility filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).mean().values
    atr_ma_14_aligned = align_htf_to_ltf(prices, df_4h, atr_ma_14)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(140, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(atr_ma_14_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility (chop) - require ATR > 1.5x MA
        vol_filter = atr_14[i] > 1.5 * atr_ma_14_aligned[i]
        
        if position == 0:
            # LONG: RSI oversold (<30) + 12h uptrend + volatility filter
            if rsi[i] < 30 and close[i] > ema_200_12h_aligned[i] and vol_filter:
                signals[i] = 0.20
                position = 1
            # SHORT: RSI overbought (>70) + 12h downtrend + volatility filter
            elif rsi[i] > 70 and close[i] < ema_200_12h_aligned[i] and vol_filter:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or trend reversal
            if rsi[i] > 70 or close[i] < ema_200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or trend reversal
            if rsi[i] < 30 or close[i] > ema_200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals