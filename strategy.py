#!/usr/bin/env python3
"""
6h_1d_keltner_breakout_trend
Hypothesis: 6-hour Keltner breakout with 1-day trend filter (EMA50) and volume confirmation.
In bull markets: buy breakouts above upper Keltner band in uptrend.
In bear markets: sell breakdowns below lower Keltner band in downtrend.
Uses volatility-adjusted bands (ATR-based) to avoid false signals in low volatility.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

name = "6h_1d_keltner_breakout_trend"
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
    
    # Get daily data for EMA trend and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day EMA50 for trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR(14) for Keltner bands
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Keltner bands (20-period EMA ± 2*ATR)
    keltner_mid = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = keltner_mid + 2 * atr
    keltner_lower = keltner_mid - 2 * atr
    
    # Align indicators to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or 
            np.isnan(keltner_lower_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price above upper Keltner band, uptrend (price > EMA50), volume confirmation
        if (close[i] > keltner_upper_aligned[i] and 
            close[i] > ema50_aligned[i] and 
            vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below lower Keltner band, downtrend (price < EMA50), volume confirmation
        elif (close[i] < keltner_lower_aligned[i] and 
              close[i] < ema50_aligned[i] and 
              vol_confirm[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or volatility collapse
        elif position == 1 and (close[i] < ema50_aligned[i] or atr[i] < 0.5 * atr[i-1] if i > 0 else False):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema50_aligned[i] or atr[i] < 0.5 * atr[i-1] if i > 0 else False):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals