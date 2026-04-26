#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_TrendFollowing
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) identifies adaptive trend,
RSI(14) filters overextended entries, and Choppiness Index (CHOP) avoids whipsaws in ranging markets.
This combination captures strong trends while minimizing trades during consolidation, reducing fee drag.
Works in bull markets (trend following) and bear markets (avoids false signals in chop, allows short signals when trend down).
Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag.
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
    
    # Pre-compute hourly filter (optional: avoid low liquidity hours if needed)
    # Using UTC 0-24 for simplicity on 1d; can adjust if needed
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA20 on 1w for HTF trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate KAMA on close (1d timeframe)
    # KAMA parameters: ER period=10, fast=2, slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will compute correctly below
    # Recompute volatility properly: sum of absolute changes over ER_period
    er_period = 10
    volatility_sum = np.zeros_like(close)
    for i in range(er_period, len(close)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    # Avoid division by zero
    volatility_sum[volatility_sum == 0] = 1e-10
    er = change / volatility_sum
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) - using 14-period
    chop_period = 14
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    atr = np.array(atr_list)
    # Sum of ATR over chop_period
    sum_atr = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    # Highest high and lowest low over chop_period
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl[range_hl == 0] = 1e-10
    chop = 100 * np.log10(sum_atr / range_hl) / np.log10(chop_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for longest indicator
    start_idx = max(30, rsi_period, chop_period, 20)  # KAMA slow=30, RSI=14, CHOP=14, EMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        close_val = close[i]
        
        # Trend filter: price above/below KAMA and aligned with 1w EMA20
        uptrend = (close_val > kama_val) and (close_val > ema_20_1w_val)
        downtrend = (close_val < kama_val) and (close_val < ema_20_1w_val)
        
        # Chop filter: avoid ranging markets (CHOP > 61.8 = choppy)
        not_choppy = chop_val < 61.8
        
        # RSI filter: avoid extremes for entry, allow exit on reversion
        rsi_overbought = rsi_val > 70
        rsi_oversold = rsi_val < 30
        
        if position == 0:
            # Long: uptrend, not choppy, RSI not overbought (avoid buying strength)
            if uptrend and not_choppy and not rsi_overbought:
                signals[i] = 0.25
                position = 1
            # Short: downtrend, not choppy, RSI not oversold (avoid selling weakness)
            elif downtrend and not_choppy and not rsi_oversold:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold
            signals[i] = 0.25
            # Exit: trend reversal or choppy market or RSI overbought (take profit)
            if not uptrend or chop_val >= 61.8 or rsi_overbought:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold
            signals[i] = -0.25
            # Exit: trend reversal or choppy market or RSI oversold (take profit)
            if not downtrend or chop_val >= 61.8 or rsi_oversold:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_TrendFollowing"
timeframe = "1d"
leverage = 1.0