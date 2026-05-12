#!/usr/bin/env python3
"""
6h Pivot Reversal with Volume Confirmation
Uses daily pivot points (PP, R1, S1, R2, S2) for mean reversion in ranging markets.
Long when price touches S2 with bullish divergence on RSI(2); short when touches R2 with bearish divergence.
Uses 1d trend filter (price vs EMA50) to avoid counter-trend trades in strong trends.
Volume spike confirms reversal intent. Targets 15-25 trades/year.
Works in bull/bear: mean reversion in ranges, trend filter avoids whipsaws in trends.
"""
name = "6h_Pivot_Reversal_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

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
    
    # === DAILY DATA FOR PIVOTS AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot points
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 6h INDICATORS ===
    # RSI(2) for short-term momentum
    rsi_period = 2
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price at S2 with bullish RSI divergence, above daily EMA50 (not in strong downtrend)
            if (low[i] <= s2_aligned[i] and 
                rsi_values[i] < 30 and  # Oversold
                close[i] > ema50_1d_aligned[i] and  # Not in strong downtrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R2 with bearish RSI divergence, below daily EMA50 (not in strong uptrend)
            elif (high[i] >= r2_aligned[i] and 
                  rsi_values[i] > 70 and  # Overbought
                  close[i] < ema50_1d_aligned[i] and  # Not in strong uptrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price reaches pivot or RSI overbought
            if (close[i] >= pivot_aligned[i]) or (rsi_values[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or RSI oversold
            if (close[i] <= pivot_aligned[i]) or (rsi_values[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals