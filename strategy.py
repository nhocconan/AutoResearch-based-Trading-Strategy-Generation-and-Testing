#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies the dominant trend efficiently with low lag. 
RSI(14) provides momentum confirmation to avoid false signals in sideways markets. 
Choppiness Index (CHOP) acts as a regime filter: we only take trend-following signals when CHOP < 45 (trending regime) 
and mean-reversion signals when CHOP > 55 (ranging regime). This dual-regime approach allows the strategy to 
profit in both bull/bear trends and sideways markets. 1d timeframe ensures low trade frequency (target: 15-25 trades/year) 
to minimize fee drag. Uses weekly EMA50 as higher timeframe trend filter to avoid counter-trend entries.
Discrete sizing (0.0, ±0.25) minimizes churn. Stoploss implemented via trend reversal signal.
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
    
    # Get 1d data for KAMA, RSI, CHOP calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA (ER=10, Fast=2, Slow=30)
    close_1d = pd.Series(df_1d['close'].values)
    change = abs(close_1d.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d.iloc[9]  # seed
    for i in range(10, len(close_1d)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama.iloc[i-1])
    kama_1d = kama.values
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1d RSI(14)
    delta = close_1d.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Choppiness Index (CHOP)
    atr_1d = pd.Series(np.maximum(np.maximum(df_1d['high'] - df_1d['low'], 
                                             abs(df_1d['high'] - df_1d['close'].shift(1))), 
                                 abs(df_1d['low'] - df_1d['close'].shift(1)))).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_1d.sum() / (max_high - min_low)) / np.log10(14)
    chop_1d = chop.values
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Regime filter: CHOP < 45 = trending, CHOP > 55 = ranging
            if chop_val < 45:
                # Trending regime: follow KAMA direction with RSI confirmation
                long_entry = (curr_close > kama_val) and (rsi_val > 50)
                short_entry = (curr_close < kama_val) and (rsi_val < 50)
            elif chop_val > 55:
                # Ranging regime: mean reversion at extremes
                long_entry = (curr_close < kama_val) and (rsi_val < 30)
                short_entry = (curr_close > kama_val) and (rsi_val > 70)
            else:
                # Neutral chop: no entries
                long_entry = False
                short_entry = False
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: trend reversal (price < KAMA) OR RSI overextended in ranging market
            if (curr_close < kama_val) or (chop_val > 55 and rsi_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: trend reversal (price > KAMA) OR RSI overextended in ranging market
            if (curr_close > kama_val) or (chop_val > 55 and rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0