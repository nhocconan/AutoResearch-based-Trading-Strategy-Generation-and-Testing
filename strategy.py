#!/usr/bin/env python3
"""
Experiment #007: 6h TRIX(21) Momentum + KAMA(21) Trend + Volume + 1d Pivot

HYPOTHESIS: Momentum-based strategy using TRIX on 6h, filtered by KAMA(21).

WHY IT SHOULD WORK IN BULL + BEAR + RANGE:
- Bull: TRIX > 0 + price > KAMA + vol spike = strong momentum continuation
- Bear: TRIX < 0 + price < KAMA + vol spike = strong momentum continuation
- Range: TRIX crossing zero + KAMA filter = avoid whipsaws in choppy markets

TRIX captures cyclical momentum at 6h (4x per day = ~4 TRIX cycles per day).
KAMA(21) smooths noise better than SMA/EMA, adapts to volatility.
1d pivot adds structural confirmation (support/resistance).

ENTRY RULES:
- LONG: TRIX crosses above 0 + price > KAMA(21) + vol > 1.5x MA
- SHORT: TRIX crosses below 0 + price < KAMA(21) + vol > 1.5x MA

TARGET: 50-150 total trades over 4 years (12-37/year).
SIZE: 0.28 (28% of capital).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_trix_kama_vol_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_trix(prices, period=21):
    """
    TRIX (Triple EMA) - momentum oscillator
    TRIX = rate of change of triple EMA
    Positive = bullish momentum, Negative = bearish momentum
    Zero line crossover = momentum shift
    """
    n = len(prices)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA calculation
    ema1 = pd.Series(prices).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX = rate of change of triple EMA (100x for readability)
    trix = np.full(n, np.nan)
    for i in range(period * 3, n):
        if ema3[i - 1] != 0:
            trix[i] = 10000 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]
    
    return trix

def calculate_kama(prices, period=14, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    Adapts to market volatility - faster in trending, slower in choppy
    """
    n = len(prices)
    if n < period:
        return np.full(n, np.nan)
    
    close = np.asarray(prices) if isinstance(prices, np.ndarray) else prices.values
    
    # Efficiency Ratio (ER)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i - period + 1:i + 1])))
    
    er = np.zeros(n)
    valid_idx = volatility > 0
    er[valid_idx] = direction[valid_idx - period] / volatility[valid_idx]
    er[:period] = 0
    
    # Smoothing constant
    sc = np.zeros(n)
    fast_const = 2 / (fast + 1)
    slow_const = 2 / (slow + 1)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_const - slow_const) + slow_const) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[period] = np.mean(close[:period])
    
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_pivot_levels(high, low, close, period=1):
    """
    Standard pivot points (Daily/HTF pivots)
    Pivot = (H + L + C) / 3
    R1 = 2 * Pivot - L, S1 = 2 * Pivot - H
    R2 = Pivot + (H - L), S2 = Pivot - (H - L)
    """
    n = len(close)
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend direction
    kama_1d = calculate_kama(df_1d['close'].values, period=21, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # 1d pivot levels
    pivot_1d, r1_1d, r2_1d, s1_1d, s2_1d = calculate_pivot_levels(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === Local 6h indicators ===
    trix = calculate_trix(close, period=21)
    kama_local = calculate_kama(close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 250  # TRIX needs 3*21=63 + buffer, plus vol MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(trix[i]) or np.isnan(kama_local[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TRIX MOMENTUM ===
        trix_value = trix[i]
        prev_trix = trix[i - 1]
        
        # TRIX crossover detection
        trix_cross_up = (prev_trix < 0) and (trix_value >= 0)
        trix_cross_down = (prev_trix > 0) and (trix_value <= 0)
        
        # === HTF TREND (1d KAMA) ===
        htf_trend_up = close[i] > kama_1d_aligned[i]
        htf_trend_down = close[i] < kama_1d_aligned[i]
        
        # === LOCAL TREND (6h KAMA) ===
        local_trend_up = close[i] > kama_local[i]
        local_trend_down = close[i] < kama_local[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === 1d PIVOT PROXIMITY (within 0.5% of pivot = at decision point) ===
        pivot_dist = abs(close[i] - pivot_aligned[i]) / pivot_aligned[i] if pivot_aligned[i] > 0 else 1.0
        near_pivot = pivot_dist < 0.005
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX crosses up + HTF trend up + local trend up + vol spike ===
            if trix_cross_up and htf_trend_up and local_trend_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: TRIX crosses down + HTF trend down + local trend down + vol spike ===
            if trix_cross_down and htf_trend_down and local_trend_down and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS: exit if momentum reverses ===
        if in_position:
            if position_side > 0:
                # Exit if TRIX turns negative
                if trix_value < 0:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if price breaks below S1 pivot
                if s1_aligned[i] > 0 and low[i] < s1_aligned[i]:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Exit if TRIX turns positive
                if trix_value > 0:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if price breaks above R1 pivot
                if r1_aligned[i] > 0 and high[i] > r1_aligned[i]:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 6 bars to avoid fee churn on 6h ===
        if in_position and (i - entry_bar) < 6:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals