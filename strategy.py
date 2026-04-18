#!/usr/bin/env python3
"""
4h_RSI_Reversal_RangeFilter
Hypothesis: In range-bound markets (BTC/ETH 2025-2026), RSI extremes at Bollinger Band levels with volume confirmation provide mean-reversion entries. Uses 12h EMA trend filter to avoid counter-trend trades. Designed for low frequency (<40 trades/year) with defined risk via Bollinger Band exit.
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
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) for range definition and entry levels
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_mid = bb_mid.values
    bb_upper = bb_upper.values
    bb_lower = bb_lower.values
    
    # RSI(14) for overbought/oversold signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Get 12h data for EMA trend filter (avoid counter-trend trades)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema34_12h[33] = np.mean(close_12h[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema34_12h[i] = close_12h[i] * alpha + ema34_12h[i-1] * (1 - alpha)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Choppiness filter: only trade when market is ranging (BB width < 50th percentile)
    bb_width = bb_upper - bb_lower
    bb_width_pct = np.full(n, np.nan)
    for i in range(50, n):
        bb_width_pct[i] = np.percentile(bb_width[max(0, i-50):i+1], 50)
    ranging = bb_width < bb_width_pct
    
    # Align 12h EMA to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(rsi[i]) or np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ranging[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold at lower BB with volume and ranging market
            if (rsi[i] < 30 and close[i] <= bb_lower[i] and vol_confirm[i] and 
                ranging[i] and close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought at upper BB with volume and ranging market
            elif (rsi[i] > 70 and close[i] >= bb_upper[i] and vol_confirm[i] and 
                  ranging[i] and close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to midline or price reaches upper BB
            if (rsi[i] > 50 or close[i] >= bb_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to midline or price reaches lower BB
            if (rsi[i] < 50 or close[i] <= bb_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Reversal_RangeFilter"
timeframe = "4h"
leverage = 1.0