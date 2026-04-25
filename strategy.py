#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum confirmation and Choppiness Index(14) for regime filtering.
Enter long when KAMA turns up, RSI > 50, and CHOP > 61.8 (ranging market). Enter short when
KAMA turns down, RSI < 50, and CHOP > 61.8. This strategy targets mean reversion in ranging
markets with trend confirmation, designed to work in both bull and bear regimes by avoiding
strong trends (CHOP < 38.2) where mean reversion fails. Uses discrete position sizing (0.25)
to minimize fee drag. Target: 15-25 trades/year on BTC/ETH/SOL.
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
    
    # Get weekly data for HTF trend filter (optional but adds robustness)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for HTF trend context
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === KAMA Calculation (ER=10, Fast=2, Slow=30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Fix alignment: volatility needs to be same length as change
    volatility = pd.Series(volume).rolling(window=10, min_periods=1).sum().values  # placeholder, recalc correctly below
    
    # Recalculate volatility correctly: sum of |close[i] - close[i-1]| over 10 periods
    price_changes = np.abs(np.diff(close))
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        start = max(0, i-9)
        volatility[i] = np.sum(price_changes[start:i+1])
    
    # Avoid division by zero
    er = np.divide(change, volatility[10:], out=np.zeros_like(change), where=volatility[10:]!=0)
    # Pad er to match close length
    er_padded = np.zeros(n)
    er_padded[10:] = er
    
    # Smoothing constants
    fast_sc = 2/(2+1)
    slow_sc = 2/(30+1)
    sc = er_padded * (fast_sc - slow_sc) + slow_sc
    sc = sc * sc  # square as per KAMA formula
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) Calculation ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match close length (first 14 values are NaN)
    rsi_padded = np.full(n, 50.0)  # default to neutral
    rsi_padded[14:] = rsi
    
    # === Choppiness Index(14) Calculation ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(14, n):
        if tr_sum[i] > 0 and hh[i] > ll[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # default to middle
    # First 14 values
    chop[:14] = 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_padded[i]) or
            np.isnan(chop[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA direction: comparing current vs previous
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI levels
        rsi_above_50 = rsi_padded[i] > 50
        rsi_below_50 = rsi_padded[i] < 50
        
        # Chop regime: > 61.8 = ranging (good for mean reversion)
        chop_ranging = chop[i] > 61.8
        chop_trending = chop[i] < 38.2  # avoid strong trends
        
        if position == 0:
            # Long: KAMA up, RSI > 50, ranging market
            if kama_up and rsi_above_50 and chop_ranging:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, ranging market
            elif kama_down and rsi_below_50 and chop_ranging:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold until KAMA reverses or chop becomes too trending
            signals[i] = 0.25
            if not kama_up or chop_trending:  # exit on KAMA down or strong trend
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold until KAMA reverses or chop becomes too trending
            signals[i] = -0.25
            if not kama_down or chop_trending:  # exit on KAMA up or strong trend
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0