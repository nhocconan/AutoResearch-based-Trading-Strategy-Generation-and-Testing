#!/usr/bin/env python3
"""
6h_RSI2_MeanReversion_1dTrendFilter_v1
Hypothesis: On 6h timeframe, use 2-period RSI for extreme mean reversion signals (RSI2<10 for long, RSI2>90 for short) filtered by 1d EMA50 trend to avoid counter-trend trades. This captures short-term reversals within the dominant daily trend, working in both bull and bear markets by only trading with the 1d trend. Targets 12-25 trades/year via strict RSI2 extremes + trend filter, minimizing fee drag while maintaining edge.
"""

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
    
    # Calculate RSI(2) - very short period for extreme mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 2
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi2 = 100 - (100 / (1 + rs))
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI2 period + EMA50 warmup
    start_idx = max(10, 50)  # RSI2 needs ~10 bars for stability, EMA50 needs 50
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for mean reversion entries with trend filter
            # Long: RSI2 < 10 (extremely oversold) + price above 1d EMA50 (uptrend)
            long_entry = (rsi2[i] < 10) and (close_val > ema_50_1d_aligned[i])
            # Short: RSI2 > 90 (extremely overbought) + price below 1d EMA50 (downtrend)
            short_entry = (rsi2[i] > 90) and (close_val < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when RSI2 reverts to midpoint (50) or touches VWAP-ish level
            # Use simple mean reversion exit: RSI2 > 50
            if rsi2[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when RSI2 reverts to midpoint (50)
            if rsi2[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI2_MeanReversion_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0