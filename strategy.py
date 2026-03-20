#!/usr/bin/env python3
"""
EXPERIMENT #003 - RSI Mean Reversion + SMA200 Trend Filter (1h)
================================================================
Hypothesis: RSI extremes combined with SMA200 trend filter will capture pullbacks 
in strong trends while avoiding counter-trend trades. This differs from Supertrend 
by entering on mean reversion within trends rather than trend breakouts.

Why this should beat Supertrend 4h (Sharpe=0.197):
- Enters on pullbacks (better entry prices) vs breakout entries
- SMA200 filter avoids counter-trend trades in ranging markets
- 1h timeframe = more opportunities than 4h while avoiding 5m/15m noise
- Discrete signal levels minimize churning costs

Key implementation:
- RSI(14) < 30 + price > SMA200 → Long (oversold in uptrend)
- RSI(14) > 70 + price < SMA200 → Short (overbought in downtrend)
- Signal = 0.0 when no clear setup (avoids whipsaw costs)
- Position size = 0.35 (same as baseline for DD control)
"""

import numpy as np
import pandas as pd

name = "rsi_sma200_1h_v1"
timeframe = "1h"
leverage = 1.0


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    n = len(close)
    
    # Calculate SMA(200) with proper min_periods
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Calculate RSI(14) with proper formula
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Rolling mean of gains and losses
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RS and RSI
    rs = np.zeros(n)
    rsi = np.zeros(n)
    
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100.0
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
        rsi[i] = 100.0 - (100.0 / (1.0 + rs[i]))
    
    # Generate signals with discrete position sizing
    signals = np.zeros(n)
    SIZE = 0.35  # 35% position size - critical for drawdown control
    
    # RSI thresholds
    RSI_OVERSOLD = 30.0
    RSI_OVERBOUGHT = 70.0
    
    # Only trade after both indicators are valid
    first_valid = max(200, 14)
    
    for i in range(first_valid, n):
        # Check for valid SMA200
        if np.isnan(sma200[i]):
            signals[i] = 0.0
            continue
        
        # Long setup: RSI oversold AND price above SMA200 (uptrend)
        if rsi[i] < RSI_OVERSOLD and close[i] > sma200[i]:
            signals[i] = SIZE
        
        # Short setup: RSI overbought AND price below SMA200 (downtrend)
        elif rsi[i] > RSI_OVERBOUGHT and close[i] < sma200[i]:
            signals[i] = -SIZE
        
        # Otherwise flat (avoid whipsaw costs)
        else:
            signals[i] = 0.0
    
    return signals