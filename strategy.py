#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily data
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 0
        else:
            er[i] = change[i] / (volatility + 1e-10) if volatility > 0 else 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    # Align KAMA to 1d timeframe (already daily, but align for safety)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get daily data for RSI calculation
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on daily data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    # Align RSI to 1d timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate volatility ratio for regime filter (ATR-based)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Volatility regime: high volatility when ATR > 1.5 * ATR(50)
    atr_ma = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    high_vol = atr > 1.5 * atr_ma
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # KAMA direction: price above KAMA = bullish, below = bearish
        kama_bullish = price > kama_1d_aligned[i]
        kama_bearish = price < kama_1d_aligned[i]
        
        # Weekly EMA200 trend filter
        uptrend = price > ema_200_1w_aligned[i]
        downtrend = price < ema_200_1w_aligned[i]
        
        # RSI conditions: avoid extremes, look for momentum
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        rsi_momentum_up = rsi_1d_aligned[i] > 50
        rsi_momentum_down = rsi_1d_aligned[i] < 50
        
        # Volatility filter: avoid extreme volatility periods
        vol_filter = not high_vol[i]
        
        if position == 0:
            # Long entry: price above KAMA, above weekly EMA200, RSI momentum up, not overbought
            if (kama_bullish and uptrend and rsi_momentum_up and 
                rsi_not_overbought and vol_filter):
                signals[i] = size
                position = 1
            # Short entry: price below KAMA, below weekly EMA200, RSI momentum down, not oversold
            elif (kama_bearish and downtrend and rsi_momentum_down and 
                  rsi_not_oversold and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price below KAMA or trend turns bearish
            if price < kama_1d_aligned[i] or price < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price above KAMA or trend turns bullish
            if price > kama_1d_aligned[i] or price > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_EMA200_RSI_Momentum_VolFilter"
timeframe = "1d"
leverage = 1.0