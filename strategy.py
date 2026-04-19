#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(close - np.roll(close, er_len))
    change[:er_len] = 0
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    volatility_pd = pd.Series(volatility)
    volatility_rolling = volatility_pd.rolling(window=er_len, min_periods=er_len).sum().values
    er = np.where(volatility_rolling != 0, change / volatility_rolling, 0)
    er[:er_len] = 0
    
    # Smoothing constant for KAMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (already aligned since we calculated on daily)
    kama_aligned = kama
    
    # 1w EMA200 for weekly trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_len, 200, 14, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # KAMA trend: price above/below KAMA
        price_above_kama = price > kama_aligned[i]
        price_below_kama = price < kama_aligned[i]
        
        # Weekly trend filter: price above/below weekly EMA200
        weekly_uptrend = price > ema200_1w_aligned[i]
        weekly_downtrend = price < ema200_1w_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price above KAMA + weekly uptrend + volume
            if price_above_kama and weekly_uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + weekly downtrend + volume
            elif price_below_kama and weekly_downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA or weekly trend turns down
            if price < kama_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA or weekly trend turns up
            if price > kama_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals