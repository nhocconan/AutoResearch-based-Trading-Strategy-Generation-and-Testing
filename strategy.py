#!/usr/bin/env python3
"""
EXPERIMENT #008 - DEMA Crossover 4h with Volatility Filter
==========================================================
Hypothesis: DEMA (Double EMA) reduces lag compared to standard EMA, providing 
earlier trend entries while maintaining smoothness. Combined with 4h timeframe 
(proven in exp#001) and ATR volatility filter to avoid choppy markets.

Key improvements over baseline:
- DEMA has less lag than EMA → earlier trend capture
- ATR volatility filter → avoid trading in low-vol chop (reduces whipsaws)
- Same conservative position sizing (0.35) to control DD
- Discrete signal levels to minimize churning costs
- Proper min_periods on all rolling calculations

Why this might beat Supertrend 4h (Sharpe=0.197):
- DEMA responds faster to trend changes than Supertrend's ATR-based stops
- Volatility filter adds regime detection (don't trade when ATR too low)
- 4h timeframe already proven to work well for crypto trends
"""

import numpy as np
import pandas as pd

name = "dema_4h_v1"
timeframe = "4h"
leverage = 1.0


def calculate_dema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Double Exponential Moving Average (DEMA)"""
    ema1 = pd.Series(prices).ewm(span=period, adjust=False, min_periods=period).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
    dema = 2 * ema1 - ema2
    return dema


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Average True Range with proper state tracking"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # DEMA parameters (faster than EMA 21/55 baseline)
    dema_fast_period = 8
    dema_slow_period = 21
    
    # Calculate DEMA lines
    dema_fast = calculate_dema(close, dema_fast_period)
    dema_slow = calculate_dema(close, dema_slow_period)
    
    # ATR for volatility filter
    atr_period = 14
    atr = calculate_atr(high, low, close, atr_period)
    
    # ATR percentile for volatility regime detection
    # Only trade when ATR is above its 20-period median (avoid low-vol chop)
    atr_median = pd.Series(atr).rolling(window=20, min_periods=20).median().values
    
    # Generate signals with discrete position sizing
    signals = np.zeros(n)
    SIZE = 0.35  # 35% position size - critical for drawdown control
    
    # Minimum period before we can generate signals
    min_period = max(dema_slow_period, atr_period, 20)
    
    for i in range(min_period, n):
        # Check if we have valid data
        if np.isnan(dema_fast[i]) or np.isnan(dema_slow[i]) or np.isnan(atr[i]) or np.isnan(atr_median[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above recent median
        # This avoids choppy low-volatility periods where crossovers whipsaw
        vol_filter = atr[i] >= atr_median[i]
        
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # DEMA crossover logic
        if dema_fast[i] > dema_slow[i]:
            signals[i] = SIZE
        elif dema_fast[i] < dema_slow[i]:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals