#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation (long when RSI>50, short when RSI<50), and Choppiness Index
regime filter to avoid whipsaws in sideways markets (only trade when CHOP < 61.8 = trending).
KAMA adapts to market noise, reducing false signals in choppy conditions. Combined with RSI
and chop filter, this should produce ~15-25 trades/year with high win rate in both bull and bear markets.
Discrete position sizing: 0.25 long/short. Max 0.30 magnitude.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (stronger regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on daily close (ER=10, Fast=2, Slow=30)
    close_1d = close  # already daily
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])).reshape(-1, 1), axis=0)  # temporary fix
    # Recompute volatility properly: sum of absolute changes over ER period
    er_period = 10
    volatility_sum = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close_1d[i-er_period:i+1])))
    # Avoid division by zero
    volatility_sum[volatility_sum == 0] = 1e-10
    er = change / volatility_sum
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily (no alignment needed as same TF, but for consistency)
    kama_aligned = kama  # same timeframe
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14)
    chop_period = 14
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(np.diff(close_1d))
    tr2 = np.abs(np.diff(high))
    tr3 = np.abs(np.diff(low))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])  # align length
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(chop_period)
    # For first chop_period-1 values, set to 50 (neutral)
    chop[:chop_period-1] = 50
    
    # Get 1w EMA20 for stronger trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    uptrend_1w = close_1d > ema_20_1w_aligned  # daily close above weekly EMA20
    downtrend_1w = close_1d < ema_20_1w_aligned  # daily close below weekly EMA20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for KAMA stability, 14 for RSI/CHOP)
    start_idx = max(30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, CHOP < 61.8 (trending), and weekly uptrend
            if (close_1d[i] > kama_aligned[i] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and 
                uptrend_1w[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, CHOP < 61.8 (trending), and weekly downtrend
            elif (close_1d[i] < kama_aligned[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and 
                  downtrend_1w[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price < KAMA OR RSI < 40 (momentum loss) OR CHOP > 61.8 (chop regime) OR weekly trend change
            if (close_1d[i] < kama_aligned[i] or 
                rsi[i] < 40 or 
                chop[i] > 61.8 or 
                not uptrend_1w[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price > KAMA OR RSI > 60 (momentum loss) OR CHOP > 61.8 (chop regime) OR weekly trend change
            if (close_1d[i] > kama_aligned[i] or 
                rsi[i] > 60 or 
                chop[i] > 61.8 or 
                not downtrend_1w[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0