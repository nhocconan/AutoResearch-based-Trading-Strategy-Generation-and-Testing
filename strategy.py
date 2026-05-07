# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "12h_TripleBarrier_1dTrend_Volume"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 12h Bollinger Bands (20, 2) for mean reversion signals
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = ma_20 + 2 * std_20
    lower_band = ma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA and BB
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(ma_20[i]) or np.isnan(std_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when volatility is elevated
        vol_condition = atr_14_1d_aligned[i] > np.nanmean(atr_14_1d_aligned[:i+1]) * 1.2
        
        if position == 0:
            # Long: price below lower BB with daily uptrend and high volatility
            if close[i] < lower_band[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price above upper BB with daily downtrend and high volatility
            elif close[i] > upper_band[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back to middle or trend reversal
            if close[i] > ma_20[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back to middle or trend reversal
            if close[i] < ma_20[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Bollinger Band mean reversion with 1d trend and volatility filter
# - In ranging markets, price reverts to mean at Bollinger Band extremes
# - Trend filter ensures we only take mean reversion trades in direction of higher timeframe trend
# - Volatility filter avoids low volatility periods where mean reversion fails
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Uses Bollinger Bands (20,2) on 12h for entry, EMA(50) on 1d for trend, ATR(14) on 1d for volatility
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volatility confirmation ensures trades occur when mean reversion is most effective
# - Simple 3-condition logic prevents overtrading and parameter sensitivity