#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_MeanReversion
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for overbought/oversold conditions, and weekly EMA(34) as a higher-timeframe trend filter.
Enter long when price > KAMA, RSI < 30 (oversold), and weekly trend is up.
Enter short when price < KAMA, RSI > 70 (overbought), and weekly trend is down.
Exit on opposite signal or when RSI reverts to mean (40-60 range).
Uses discrete position sizing (0.25) to limit fee drag. Designed to work in both bull and bear markets
by combining trend-following (KAMA) with mean-reversion (RSI extremes) and HTF alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # 1d data for KAMA and RSI (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA(10, 2, 30) - ER based on 10-period, fast=2, slow=30
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0, keepdims=True).flatten()
    # Correct ER calculation: sum of absolute changes over sum of absolute price movements
    er_num = np.zeros_like(close_1d)
    er_den = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        er_num[i] = er_num[i-1] + np.abs(close_1d[i] - close_1d[i-10]) if i >= 10 else np.abs(close_1d[i] - close_1d[0])
        er_den[i] = er_den[i-1] + np.sum(np.abs(np.diff(close_1d[max(0,i-9):i+1]))) if i >= 9 else np.sum(np.abs(np.diff(close_1d[0:i+1])))
    er = np.where(er_den != 0, er_num / er_den, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to lower timeframe (though primary is 1d, we keep alignment for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1w data for EMA34 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d KAMA/RSI (30) and 1w EMA34 (34)
    start_idx = max(30, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_kama = kama_aligned[i]
        curr_rsi = rsi_aligned[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        
        # Trend filter from weekly EMA34
        uptrend_1w = curr_close > curr_ema_1w
        downtrend_1w = curr_close < curr_ema_1w
        
        if position == 0:
            # Look for entry signals
            # Long: price > KAMA (bullish bias), RSI < 30 (oversold), weekly uptrend
            long_entry = (curr_close > curr_kama) and (curr_rsi < 30) and uptrend_1w
            # Short: price < KAMA (bearish bias), RSI > 70 (overbought), weekly downtrend
            short_entry = (curr_close < curr_kama) and (curr_rsi > 70) and downtrend_1w
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if RSI reverts to mean (40-60) or weekly trend turns down
            if (curr_rsi >= 40 and curr_rsi <= 60) or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if RSI reverts to mean (40-60) or weekly trend turns up
            if (curr_rsi >= 40 and curr_rsi <= 60) or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0