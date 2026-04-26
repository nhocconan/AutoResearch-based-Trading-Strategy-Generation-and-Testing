#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v2
Hypothesis: On daily timeframe, KAMA direction (trend) + RSI(14) extreme + Choppiness Index regime filter captures sustained moves while avoiding whipsaw in choppy markets. Works in bull/bear by taking directional entries only when aligned with KAMA trend and RSI confirms momentum, while chop filter avoids false signals in ranging markets. Target: 15-25 trades/year.
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w < 50):
        return np.zeros(n)
    
    # 1w EMA34 for HTF trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # KAMA direction (primary trend)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # needs correction
    # Recalculate properly
    er = np.zeros(n)
    for i in range(10, n):
        change_val = np.abs(close[i] - close[i-10])
        volatility_val = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if volatility_val > 0:
            er[i] = change_val / volatility_val
        else:
            er[i] = 1.0
    er[0:10] = 1.0
    
    # Smoothing constants
    fast_sc = 2/(2+1)
    slow_sc = 2/(30+1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[0:14] = 50.0  # neutral until enough data
    
    # Choppiness Index (14)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros(n)
    for i in range(14, n):
        if atr14[i] > 0 and highest_high14[i] > lowest_low14[i]:
            chop[i] = 100 * np.log10(atr14[i] / (highest_high14[i] - lowest_low14[i])) / np.log10(14)
        else:
            chop[i] = 50.0
    chop[0:14] = 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of KAMA calc (10), RSI (14), Chop (14), HTF EMA (34)
    start_idx = max(10, 14, 14, 34)
    
    for i in range(start_idx, n):
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        htf_trend = ema34_1w_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or np.isnan(htf_trend)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filters
        kama_up = close_val > kama_val
        kama_down = close_val < kama_val
        htf_up = close_val > htf_trend
        htf_down = close_val < htf_trend
        
        # Momentum filter
        rsi_overbought = rsi_val > 70
        rsi_oversold = rsi_val < 30
        rsi_momentum_up = rsi_val > 50  # bullish momentum
        rsi_momentum_down = rsi_val < 50  # bearish momentum
        
        # Regime filter: avoid choppy markets (chop > 61.8 = ranging)
        is_trending = chop_val < 61.8
        
        # Entry conditions: KAMA direction aligned with HTF trend + RSI momentum + trending regime
        long_entry = kama_up and htf_up and rsi_momentum_up and is_trending
        short_entry = kama_down and htf_down and rsi_momentum_down and is_trending
        
        # Exit conditions: opposite KAMA cross or RSI extreme reversal
        long_exit = False
        short_exit = False
        if position == 1:
            long_exit = (close_val < kama_val) or (rsi_val > 80)  # trend break or overextended
        elif position == -1:
            short_exit = (close_val > kama_val) or (rsi_val < 20)  # trend break or overextended
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0