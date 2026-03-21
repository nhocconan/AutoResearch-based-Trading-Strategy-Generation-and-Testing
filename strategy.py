#!/usr/bin/env python3
"""
EXPERIMENT #017 - 1h KAMA Momentum with 4h HMA Trend + ATR Stop
===============================================================
Hypothesis: KAMA adapts to market noise better than EMA/HMA during ranging periods.
Combining 4h HMA trend direction with 1h KAMA momentum crossovers should reduce
whipsaws while capturing trends. ATR trailing stop protects against reversals.

Key differences from failed experiments:
- KAMA (adaptive) instead of HMA/EMA for primary momentum
- Volume spike confirmation on entries (avoid low-liquidity traps)
- Explicit ATR trailing stoploss (signal→0 when stopped)
- Conservative position sizing (0.25 base, max 0.35)
- Discrete signal levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_kama_hma_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / volatility
    er = np.nan_to_num(er, nan=0.0)
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = np.clip(sc, 0.0, 1.0)
    
    # Initialize KAMA
    kama[:period] = np.nan
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] ** 2 * (close[i] - kama[i-1])
    
    return kama


def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_hma(close, period=21):
    """Hull Moving Average"""
    close_series = pd.Series(close)
    wma_half = close_series.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean()
    hma = (2 * wma_half - wma_full).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === LOAD 4H HTF DATA ONCE (CRITICAL - Rule 1) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(close_4h, period=21)
    hma_4h_prev = np.roll(hma_4h, 1)
    hma_4h_prev[0] = hma_4h[0]
    trend_4h = np.where(hma_4h > hma_4h_prev, 1, -1)
    
    # Align 4h trend to 1h (with proper shift for completed bars - Rule 2)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === CALCULATE 1H INDICATORS (before loop - Rule 8) ===
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume moving average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Initialize signals
    signals = np.zeros(n)
    
    # Position tracking for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Position sizing
    BASE_SIZE = 0.25  # 25% base position
    MAX_SIZE = 0.35   # 35% max position
    
    # Find first valid index (all indicators ready)
    first_valid = max(30, int(np.sqrt(21)))  # KAMA + HMA warmup
    
    for i in range(first_valid, n):
        # Skip if indicators are NaN
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Get aligned 4h trend (already shifted for completed bars)
        htf_trend = trend_4h_aligned[i]
        
        # Volume spike confirmation (1.5x average)
        vol_spike = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # === ENTRY SIGNALS ===
        if position_side == 0:
            # Long entry: 4h trend up + KAMA fast crosses above slow + volume spike
            if htf_trend == 1 and kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]:
                if vol_spike or i > first_valid + 100:  # Allow entry without volume after warmup
                    signals[i] = BASE_SIZE
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
            
            # Short entry: 4h trend down + KAMA fast crosses below slow + volume spike
            elif htf_trend == -1 and kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]:
                if vol_spike or i > first_valid + 100:
                    signals[i] = -BASE_SIZE
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
        
        # === POSITION MANAGEMENT ===
        elif position_side == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, close[i])
            
            # ATR trailing stoploss (2.5 ATR from highest)
            stop_price = highest_since_entry - 2.5 * atr[i]
            
            # Take profit at 2R (reduce to half)
            profit_target = entry_price + 2.0 * (entry_price - (entry_price - 2.5 * atr[i]))
            
            if close[i] < stop_price:
                # Stoploss hit - exit position
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            elif close[i] > profit_target and signals[i] != BASE_SIZE / 2:
                # Take profit - reduce position by half
                signals[i] = BASE_SIZE / 2
            else:
                signals[i] = BASE_SIZE
        
        elif position_side == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, close[i])
            
            # ATR trailing stoploss (2.5 ATR from lowest)
            stop_price = lowest_since_entry + 2.5 * atr[i]
            
            # Take profit at 2R (reduce to half)
            profit_target = entry_price - 2.0 * ((entry_price + 2.5 * atr[i]) - entry_price)
            
            if close[i] > stop_price:
                # Stoploss hit - exit position
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            elif close[i] < profit_target and signals[i] != -BASE_SIZE / 2:
                # Take profit - reduce position by half
                signals[i] = -BASE_SIZE / 2
            else:
                signals[i] = -BASE_SIZE
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend reverses against position
        if position_side == 1 and htf_trend == -1:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
        elif position_side == -1 and htf_trend == 1:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
    
    return signals