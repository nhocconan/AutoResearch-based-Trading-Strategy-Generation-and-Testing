#!/usr/bin/env python3
"""
1h_4d_Trend_Pullback_EMA_RSI
Hypothesis: In strong 4h trends (price > EMA50), pullbacks to EMA20 on 1h with RSI < 40 offer high-probability long entries. 
For shorts: 4h downtrend (price < EMA50), pullback to EMA20 with RSI > 60. Uses 1d volatility filter (ATR ratio) to avoid choppy markets.
Designed for low trade frequency (15-30/year) by requiring 4h trend alignment and 1h pullback confirmation. Works in bull markets via longs and bear markets via shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data once for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d ATR(14) and ATR(50) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50_1d = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr14_1d / (atr50_1d + 1e-10)  # avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 1h EMA20 and RSI(14) for entry timing
    close = prices['close'].values
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50_4h_val = ema50_4h_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        ema20_val = ema20[i]
        rsi_val = rsi[i]
        
        # Volatility filter: only trade when ATR ratio < 1.2 (low volatility)
        vol_filter = atr_ratio_val < 1.2
        
        if position == 0:
            # Long: 4h uptrend + pullback to EMA20 + RSI oversold
            if (price > ema50_4h_val and  # 4h uptrend
                price > ema20_val and     # above 1h EMA20
                price < ema20_val * 1.02 and  # within 2% of EMA20 (pullback zone)
                rsi_val < 40 and          # RSI oversold
                vol_filter):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + pullback to EMA20 + RSI overbought
            elif (price < ema50_4h_val and  # 4h downtrend
                  price < ema20_val and     # below 1h EMA20
                  price > ema20_val * 0.98 and  # within 2% of EMA20 (pullback zone)
                  rsi_val > 60 and          # RSI overbought
                  vol_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h trend breaks or RSI overbought
            if (price < ema50_4h_val or    # 4h trend broken
                rsi_val > 70):             # RSI overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h trend breaks or RSI oversold
            if (price > ema50_4h_val or    # 4h trend broken
                rsi_val < 30):             # RSI oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4d_Trend_Pullback_EMA_RSI"
timeframe = "1h"
leverage = 1.0