#!/usr/bin/env python3
"""
4h_RSI_Overbought_Sell_30min_Exit_v1
Hypothesis: RSI(14) > 70 on 4h indicates short-term exhaustion in both bull and bear markets.
Exit after fixed 30-minute hold (2 bars on 4h) to capture mean reversion without overtrading.
Filters: volume > 1.5x 20-bar average and price > 200-period EMA (avoid shorts in strong uptrends).
Works in bull (mean reversion in uptrend) and bear (short-term bounces in downtrend).
Target: ~50 trades/year, low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA200 for trend filter (avoid counter-trend shorts)
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: > 1.5x 20-bar average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, -1: short
    bars_held = 0
    size = 0.25   # Position size: 25% of capital
    
    start_idx = 200  # EMA200 warmup
    
    for i in range(start_idx, n):
        if bars_held > 0:
            bars_held -= 1
            if bars_held == 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size if position == -1 else 0.0
            continue
        
        rsi_val = rsi[i]
        ema200_val = ema200[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Short signal: RSI > 70 (overbought) + volume confirmation + not in strong uptrend
            if rsi_val > 70 and vol_conf and close[i] < ema200_val:
                signals[i] = -size
                position = -1
                bars_held = 2  # Hold for 2 bars (30 minutes on 4h)
    
    return signals

name = "4h_RSI_Overbought_Sell_30min_Exit_v1"
timeframe = "4h"
leverage = 1.0