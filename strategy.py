#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) Extreme Reversion with 4h Trend Filter and Volume Confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for trend filter (EMA50).
- Entry: RSI(2) < 10 for long, RSI(2) > 90 for short, with price > 4h EMA50 (long) or < 4h EMA50 (short), and volume > 1.5x 20-period volume MA.
- Exit: RSI(2) crosses above 50 (long exit) or below 50 (short exit).
- Rationale: In both bull and bear markets, short-term extremes revert, but only when aligned with the 4h trend to avoid counter-trend traps.
- Volume confirmation reduces false signals.
- Discrete signal size: 0.20 to control drawdown and fees.
- Target: 80-150 total trades over 4 years (20-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(2) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 2)  # Need 4h EMA50, volume MA(20), RSI(2)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_2[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI(2) < 10 (extreme oversold) AND uptrend (close > 4h EMA50) AND volume spike
            if (rsi_2[i] < 10 and close[i] > ema_50_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 (extreme overbought) AND downtrend (close < 4h EMA50) AND volume spike
            elif (rsi_2[i] > 90 and close[i] < ema_50_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI(2) crosses above 50 (mean reversion complete)
            if rsi_2[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI(2) crosses below 50 (mean reversion complete)
            if rsi_2[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_Extreme_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0