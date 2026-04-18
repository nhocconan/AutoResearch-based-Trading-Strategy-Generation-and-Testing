#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
Hypothesis: KAU (Kaufman's Adaptive Moving Average) adapts to market noise, reducing whipsaw in choppy markets while capturing trends. Combined with RSI(14) > 50 for long and < 50 for short, it filters counter-trend signals. Weekly trend filter (EMA50) ensures alignment with higher-timeframe momentum. Designed for low frequency (10-25 trades/year) to minimize fee decay while capturing sustained moves in bull and bear markets.
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
    
    # KAU (Kaufman's Adaptive Moving Average)
    # ER = Efficiency Ratio = |net change| / sum(|changes|)
    # Smooth = ER * fastest SC + (1 - ER) * slowest SC
    # SC = 2/(n+1) for EMA
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 1)))
    direction[0] = 0
    
    # Avoid division by zero
    sum_change = np.nancumsum(change)
    er = np.where(sum_change > 0, direction / sum_change, 0)
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kaufman = np.zeros_like(close)
    kaufman[0] = close[0]
    for i in range(1, n):
        kaufman[i] = kaufman[i-1] + sc[i] * (close[i] - kaufman[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need EMA50 and RSI warmup
    
    for i in range(start_idx, n):
        if np.isnan(kaufman[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kaufman_val = kaufman[i]
        rsi_val = rsi[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price > KAU and RSI > 50 and above weekly EMA50
            if price > kaufman_val and rsi_val > 50 and price > ema_50_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: price < KAU and RSI < 50 and below weekly EMA50
            elif price < kaufman_val and rsi_val < 50 and price < ema_50_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < KAU or RSI < 40 or below weekly EMA50
            if price < kaufman_val or rsi_val < 40 or price < ema_50_1w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > KAU or RSI > 60 or above weekly EMA50
            if price > kaufman_val or rsi_val > 60 or price > ema_50_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0