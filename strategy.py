#!/usr/bin/env python3
"""
EXPERIMENT #009 - KAMA Adaptive Trend + RSI Pullback + 12h Filter (1h)
=================================================================
Hypothesis: 1h primary timeframe with 12h KAMA trend filter (more stable than 4h).
Enter on RSI pullbacks when KAMA slope confirms trend direction.
Z-score filter avoids extreme mean-reversion conditions.
ATR trailing stoploss at 2.5*ATR for risk control.

Why this should work:
- KAMA adapts to volatility (better than HMA in choppy markets)
- 12h trend filter is more stable than 4h (fewer whipsaws)
- RSI pullbacks give better entry timing than breakouts
- Z-score filter avoids entering at extremes
- Conservative position sizing (0.25) controls drawdown
- 1h timeframe balances signal frequency vs noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_12h_zscore_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        # Efficiency Ratio
        if i >= er_period:
            change = abs(close[i] - close[i - er_period])
            volatility = np.sum(np.abs(np.diff(close[max(0, i-er_period):i+1])))
            er = change / volatility if volatility > 0 else 0
        else:
            er = 0
        
        # Smoothing Constant
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 
              2.0 / (slow_period + 1)) ** 2
        
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std
    return zscore.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_kama_slope(kama, lookback=5):
    """Calculate KAMA slope for trend confirmation"""
    n = len(kama)
    slope = np.zeros(n)
    for i in range(lookback, n):
        slope[i] = kama[i] - kama[i - lookback]
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1)
    df_12h = get_htf_data(prices, '12h')
    kama_12h = calculate_kama(df_12h['close'].values, 10, 2, 30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 1h indicators
    kama_1h = calculate_kama(close, 10, 2, 30)
    kama_1h_slope = calculate_kama_slope(kama_1h, 5)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    atr = calculate_atr(high, low, close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25  # Conservative position size (25% of capital)
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    min_period = 100  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN or invalid values in any indicator
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1h[i]) or 
            np.isnan(rsi[i]) or np.isnan(zscore[i]) or np.isnan(atr[i]) or 
            atr[i] == 0 or np.isnan(kama_1h_slope[i])):
            signals[i] = 0.0
            continue
        
        # 12h trend filter (primary trend direction)
        trend_12h = 1 if close[i] > kama_12h_aligned[i] else -1
        
        # 1h KAMA slope confirmation (trend momentum)
        slope_1h = 1 if kama_1h_slope[i] > 0 else -1
        
        # Z-score filter (avoid extreme mean-reversion conditions)
        zscore_valid = abs(zscore[i]) < 2.0  # Not at extreme levels
        
        # RSI pullback signal (entry on pullback in trend direction)
        rsi_signal = 0
        if trend_12h == 1:  # Uptrend - look for long on RSI pullback
            if rsi[i] < 50 and rsi[i-1] >= 50:  # RSI crossed below 50 (pullback)
                rsi_signal = 1
        else:  # Downtrend - look for short on RSI bounce
            if rsi[i] > 50 and rsi[i-1] <= 50:  # RSI crossed above 50 (bounce)
                rsi_signal = -1
        
        # Trend alignment filter (12h trend and 1h slope must agree)
        trend_aligned = (trend_12h == slope_1h)
        
        # Determine target signal
        target_signal = 0.0
        if rsi_signal != 0 and zscore_valid and trend_aligned:
            target_signal = SIZE_ENTRY * rsi_signal
        
        # Stoploss logic - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                entry_price = close[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE_ENTRY * position_side
            else:
                signals[i] = 0.0
    
    return signals