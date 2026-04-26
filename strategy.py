#!/usr/bin/env python3
"""
4h_KAMA_Direction_1dRSI_ChopFilter
Hypothesis: 4h KAMA direction filter with 1d RSI regime filter and volume confirmation. 
- Long when KAMA upward (close > KAMA) AND 1d RSI between 40-60 (neutral) AND volume > 1.5 * volume_ma(20)
- Short when KAMA downward (close < KAMA) AND 1d RSI between 40-60 AND volume > 1.5 * volume_ma(20)
- KAMA adapts to market noise, reducing whipsaws in ranging markets
- 1d RSI regime filter avoids extreme overbought/oversold conditions that often reverse in bear markets
- Volume confirmation ensures breakouts have participation
- Designed for low frequency (target 20-40 trades/year) to minimize fee drag
- Novelty: Uses KAMA's adaptive smoothing with RSI regime filter for bear market resilience
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14) for regime filter (needs completed 1d candle)
    delta = pd.Series(df_1d['close'].values).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    # Regime: 1 = neutral (40 <= RSI <= 60), 0 = extreme (avoid trading)
    regime_1d = np.where((rsi_1d_aligned >= 40) & (rsi_1d_aligned <= 60), 1, 0)
    
    # Calculate KAMA(10,2,30) on 4h chart (primary timeframe)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=1)  # 10-period sum of absolute changes
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9 (10th element)
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for KAMA, 20 for volume MA, 14 for RSI)
    start_idx = max(30, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(regime_1d[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA direction conditions with regime and volume filter
        if position == 0:
            # Long: Price above KAMA AND 1d RSI neutral AND volume spike
            if close[i] > kama[i] and regime_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA AND 1d RSI neutral AND volume spike
            elif close[i] < kama[i] and regime_1d[i] == 1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below KAMA OR 1d RSI turns extreme
            if close[i] < kama[i] or regime_1d[i] == 0:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above KAMA OR 1d RSI turns extreme
            if close[i] > kama[i] or regime_1d[i] == 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Direction_1dRSI_ChopFilter"
timeframe = "4h"
leverage = 1.0