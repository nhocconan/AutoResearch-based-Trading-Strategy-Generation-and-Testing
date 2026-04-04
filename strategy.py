#!/usr/bin/env python3
"""
exp_6850_1d_kama_rsi_chop_v1
Hypothesis: 1d KAMA trend + RSI(14) extremes + Choppiness Index regime filter.
In trending markets (CHOP < 38.2): follow KAMA direction (long if price > KAMA, short if price < KAMA).
In ranging markets (CHOP > 61.8): mean revert at RSI extremes (long if RSI < 30, short if RSI > 70).
Only trade when volume confirms (volume > 1.5x 20-day average). Uses discrete position sizing (0.25)
to minimize fee churn. Designed for low trade frequency (~10-25/year) to avoid overtrading.
Works in both bull and bear markets by adapting to regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6850_1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
KAMA_ER_PERIOD = 10
KAMA_FAST = 2
KAMA_SLOW = 30
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
CHOP_PERIOD = 14
CHOP_THRESHOLD_TRENDING = 38.2
CHOP_THRESHOLD_RANGING = 61.8
VOL_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 40  # ~1.3 months

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Calculate indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute correctly below
    # Recompute volatility properly: sum of absolute changes over ER_PERIOD
    volatility = pd.Series(close).diff().abs().rolling(window=KAMA_ER_PERIOD, min_periods=1).sum().values
    change = pd.Series(close).diff().abs().rolling(window=KAMA_ER_PERIOD, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/(KAMA_FAST+1) - 2/(KAMA_SLOW+1)) + 2/(KAMA_SLOW+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index
    atr1 = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))))
    atr_sum = atr1.rolling(window=CHOP_PERIOD, min_periods=CHOP_PERIOD).sum().values
    highest_high = pd.Series(high).rolling(window=CHOP_PERIOD, min_periods=CHOP_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=CHOP_PERIOD, min_periods=CHOP_PERIOD).min().values
    range_max_min = highest_high - lowest_low
    chop = np.where(range_max_min != 0, 100 * np.log10(atr_sum / range_max_min) / np.log10(CHOP_PERIOD), 50)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(KAMA_ER_PERIOD, RSI_PERIOD, CHOP_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if any data not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOLUME_THRESHOLD
        
        # Determine regime
        is_trending = chop[i] < CHOP_THRESHOLD_TRENDING
        is_ranging = chop[i] > CHOP_THRESHOLD_RANGING
        
        # Initialize signal
        new_signal = 0
        
        if is_trending and vol_confirmed:
            # Trend following: follow KAMA direction
            if close[i] > kama[i]:
                new_signal = SIGNAL_SIZE  # long
            elif close[i] < kama[i]:
                new_signal = -SIGNAL_SIZE  # short
        elif is_ranging and vol_confirmed:
            # Mean reversion: fade extremes at RSI levels
            if rsi[i] < RSI_OVERSOLD:
                new_signal = SIGNAL_SIZE  # long
            elif rsi[i] > RSI_OVERBOUGHT:
                new_signal = -SIGNAL_SIZE  # short
        
        # Only change position if signal differs from current
        if new_signal != position * SIGNAL_SIZE:
            signals[i] = new_signal
            if new_signal != 0:
                position = 1 if new_signal > 0 else -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                position = 0
                bars_since_entry = 0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals