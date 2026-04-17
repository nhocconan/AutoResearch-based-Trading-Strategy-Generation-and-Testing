#!/usr/bin/env python3
"""
Hypothesis: On the 4-hour timeframe, we combine the 1-day True Range (ATR) for volatility,
the 1-day RSI for mean-reversion signals, and a 1-day ADX for trend strength to filter
entries. We go long when price is below the 1-day low plus 0.5*ATR (oversold bounce) with
RSI < 30 and ADX < 25 (ranging market). We go short when price is above the 1-day high
minus 0.5*ATR (overbought rejection) with RSI > 70 and ADX < 25. Exit when RSI crosses
50 or on opposite signal. Designed for low-frequency mean reversion in ranging markets
with ~20-40 trades per year to avoid fee drag.
"""

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
    
    # Get 1d data for ATR, RSI, ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day ATR(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-day RSI(14)
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = (100 - (100 / (1 + rs))).values
    
    # Calculate 1-day ADX(14)
    plus_dm = df_1d['high'].diff()
    minus_dm = df_1d['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr_14 = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / tr_14)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / tr_14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_1d = dx.rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1-day high and low for entry levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(high_1d_aligned[i]) or
            np.isnan(low_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        rsi = rsi_1d_aligned[i]
        adx = adx_1d_aligned[i]
        high_1d_val = high_1d_aligned[i]
        low_1d_val = low_1d_aligned[i]
        
        if position == 0:
            # Long: price near 1d low (oversold bounce) in ranging market
            if price <= low_1d_val + 0.5 * atr and rsi < 30 and adx < 25:
                signals[i] = 0.25
                position = 1
            # Short: price near 1d high (overbought rejection) in ranging market
            elif price >= high_1d_val - 0.5 * atr and rsi > 70 and adx < 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 or price reaches 1d high
            if rsi >= 50 or price >= high_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 or price reaches 1d low
            if rsi <= 50 or price <= low_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_ATR_RSI_ADX_MeanReversion"
timeframe = "4h"
leverage = 1.0