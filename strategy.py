#!/usr/bin/env python3
"""
6h_Daily_Pivot_Reversal_v1
Hypothesis: In 6h timeframe, price tends to reverse at daily pivot points (PP) and support/resistance levels (S1, R1).
Go long when price touches S1 with bullish reversal candle (close > open) and RSI < 40.
Go short when price touches R1 with bearish reversal candle (close < open) and RSI > 60.
Use volatility filter (ATR < 1.5x ATR(50)) to avoid whipsaws in high volatility.
Target: 20-40 trades/year per symbol to minimize fee drag. Works in ranging and trending markets via mean reversion at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for pivot calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    
    # RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    roll_down = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily pivot levels to 6h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extreme volatility
        vol_filter = atr[i] < 1.5 * atr_ma[i]
        
        # Price touching S1 or R1 (within 0.1% tolerance)
        touch_s1 = np.abs(low[i] - s1_aligned[i]) / s1_aligned[i] < 0.001
        touch_r1 = np.abs(high[i] - r1_aligned[i]) / r1_aligned[i] < 0.001
        
        # Reversal candle: bullish (close > open) or bearish (close < open)
        bullish_candle = close[i] > prices['open'].iloc[i]
        bearish_candle = close[i] < prices['open'].iloc[i]
        
        if position == 0:
            # Long: price touches S1 with bullish candle and RSI < 40 (oversold)
            if touch_s1 and bullish_candle and rsi[i] < 40 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 with bearish candle and RSI > 60 (overbought)
            elif touch_r1 and bearish_candle and rsi[i] > 60 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches PP or RSI > 60 (overbought) or volatility filter fails
            if close[i] >= pp_aligned[i] or rsi[i] > 60 or not vol_filter:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches PP or RSI < 40 (oversold) or volatility filter fails
            if close[i] <= pp_aligned[i] or rsi[i] < 40 or not vol_filter:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Daily_Pivot_Reversal_v1"
timeframe = "6h"
leverage = 1.0