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
    
    # Load 1d and 1w data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile (20-day lookback) for regime filter
    atr_percentile = pd.Series(atr_1d).rolling(window=20, min_periods=20).quantile(0.5).values
    # Low volatility regime: current ATR < 50th percentile of recent ATR
    low_vol_regime = atr_1d < atr_percentile
    
    # Align 1d volatility regime to 6h
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    # Calculate 1-week trend using EMA crossover
    ema_fast = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_slow = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_uptrend = ema_fast > ema_slow
    
    # Align weekly trend to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    
    # Calculate 6-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(low_vol_aligned[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long in low volatility + weekly uptrend when RSI oversold
            if (low_vol_aligned[i] > 0.5 and 
                weekly_uptrend_aligned[i] > 0.5 and
                rsi[i] < 30):
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility + weekly downtrend when RSI overbought
            elif (low_vol_aligned[i] > 0.5 and 
                  weekly_uptrend_aligned[i] < 0.5 and
                  rsi[i] > 70):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when RSI returns to neutral zone
            if position == 1:
                if rsi[i] >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi[i] <= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_VolRegime_WeeklyTrend_RSI_MeanRev"
timeframe = "6h"
leverage = 1.0