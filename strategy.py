#!/usr/bin/env python3
# 4h_Keltner_RSI_Trend_Follow
# Hypothesis: In trending markets, price pulls back to the Keltner Channel middle line (EMA)
# with RSI showing exhaustion, then continues in trend direction. Works in bull via pullback longs
# and in bear via pullback shorts. Trend filter from 1d EMA50 avoids counter-trend trades.
# Uses 4h timeframe for balanced trade frequency (~20-50/year). Volume confirmation adds robustness.
# Risk control via trailing stop using ATR-based channel width.

name = "4h_Keltner_RSI_Trend_Follow"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # === 4H INDICATORS ===
    # EMA20 for Keltner middle and trend
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR for Keltner width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    keltner_middle = ema20
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr
    
    # RSI(14) for momentum exhaustion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1D TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ma * 1.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 and RSI ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(keltner_middle[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Pullback to middle in uptrend with RSI < 40 and volume
            if (close[i] > ema50_1d_aligned[i] and  # Uptrend filter
                low[i] <= keltner_middle[i] * 1.002 and  # Touched or slightly below middle
                rsi[i] < 40 and  # Momentum exhaustion
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback to middle in downtrend with RSI > 60 and volume
            elif (close[i] < ema50_1d_aligned[i] and  # Downtrend filter
                  high[i] >= keltner_middle[i] * 0.998 and  # Touched or slightly above middle
                  rsi[i] > 60 and  # Momentum exhaustion
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close below keltner lower or RSI > 70 (overbought)
            if close[i] < keltner_lower[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above keltner upper or RSI < 30 (oversold)
            if close[i] > keltner_upper[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals