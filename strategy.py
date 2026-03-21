#!/usr/bin/env python3
"""
EXPERIMENT #019 - 1h Primary with 12h HMA Trend + RSI Pullback + ATR Stop
==========================================================================
Hypothesis: 12h HMA provides stronger trend filter than 4h (fewer whipsaws).
RSI pullback entries (30-40 for long, 60-70 for short) capture better risk/reward.
ATR trailing stop (2.5x) protects capital without premature exits.
Volume filter reduces false breakouts.

Key improvements over failed experiments:
- 12h HTF (vs 4h/6h) = stronger trend, fewer reversals
- RSI range entry (not extreme) = better timing on pullbacks
- ATR stop only exits position, doesn't flip = reduces churn
- Discrete sizing (0.0, ±0.30) = minimal fee impact
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_12h_hma_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.rolling(window=period//2, min_periods=period//2).mean()
    wma2 = close_s.rolling(window=period, min_periods=period).mean()
    wma_diff = 2 * wma1 - wma2
    hma = wma_diff.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI indicator"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (CRITICAL - Rule 1)
    df_12h = get_htf_data(prices, '12h')
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h = align_htf_to_ltf(prices, df_12h, hma_12h_raw)  # auto shift(1)
    
    # Calculate 1h indicators (vectorized before loop - Rule 8)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # Volume ratio (current vs 20-bar average)
    volume_s = pd.Series(volume)
    volume_avg = volume_s.rolling(window=20, min_periods=20).mean()
    volume_ratio = (volume_s / volume_avg).values
    
    # 1h HMA for local trend confirmation
    hma_1h = calculate_hma(close, 21)
    
    # Initialize signals and position tracking
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (discrete, Rule 4)
    
    # Position state tracking for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period for all indicators
    start_idx = max(50, int(np.sqrt(21)) + 20)
    
    for i in range(start_idx, n):
        # Skip invalid data
        if np.isnan(hma_12h[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(hma_1h[i]):
            signals[i] = 0.0
            continue
        
        current_atr = atr[i]
        if current_atr <= 0 or not np.isfinite(current_atr):
            signals[i] = 0.0
            continue
        
        # Calculate trailing stop distance
        stop_distance = 2.5 * current_atr
        
        # CHECK STOPLOSS FIRST (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trail_stop = highest_since_entry - stop_distance
            
            if close[i] < trail_stop:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trail_stop = lowest_since_entry + stop_distance
            
            if close[i] > trail_stop:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # ENTRY LOGIC (only when flat)
        if position_side == 0:
            # 12h trend direction
            trend_12h = hma_12h[i] - hma_12h[i-1] if i > 0 else 0
            
            # Long entry: 12h uptrend + price above 12h HMA + RSI pullback + volume
            if (trend_12h > 0 and 
                close[i] > hma_12h[i] and 
                30 <= rsi[i] <= 45 and 
                hma_1h[i] > hma_1h[i-5] and
                volume_ratio[i] >= 0.8):
                signals[i] = SIZE
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
            
            # Short entry: 12h downtrend + price below 12h HMA + RSI bounce + volume
            elif (trend_12h < 0 and 
                  close[i] < hma_12h[i] and 
                  55 <= rsi[i] <= 70 and 
                  hma_1h[i] < hma_1h[i-5] and
                  volume_ratio[i] >= 0.8):
                signals[i] = -SIZE
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
        
        # HOLD POSITION (maintain signal, avoid churn)
        elif position_side == 1:
            signals[i] = SIZE
        elif position_side == -1:
            signals[i] = -SIZE
    
    return signals