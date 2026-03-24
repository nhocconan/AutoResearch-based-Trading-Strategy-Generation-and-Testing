#!/usr/bin/env python3
"""
Experiment #015: 6h Primary + 12h/1d HTF — Fisher Transform + Vol Spike Reversion + HMA Trend

Hypothesis: After 14 failed experiments, the pattern shows regime-switching strategies fail on 6h.
NEW APPROACH: Ehlers Fisher Transform excels at catching reversals in bear/range markets (2022, 2025).
Combined with volatility spike detection (ATR ratio) for mean reversion entries, this captures
"panic capitulation" bottoms and "euphoria" tops that simple trend strategies miss.

Key innovations vs failed strategies:
- Fisher Transform (period=9): Normalizes price to Gaussian, crosses at ±1.5 signal reversals
- Vol Spike Filter: ATR(7)/ATR(30) > 1.8 = panic/extreme vol = mean reversion opportunity
- 12h HMA intermediate trend: Smoother than 6h, faster than 1d for alignment
- 1d HMA major bias: Only trade Fisher signals aligned with major trend
- Asymmetric sizing: 0.30 for high-conviction (vol spike + Fisher + HTF), 0.20 for normal

Why this should work on 6h:
- 6h captures multi-day swings without 15m/1h noise
- Fisher catches reversals that Donchian/EMA miss in choppy markets
- Vol spike filter ensures entries only at extremes (fewer trades, higher quality)
- 12h/1d HTF prevents counter-trend trades that destroyed 2022 performance

Target: Sharpe>0.02, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_volspike_hma_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian normal distribution for clearer reversal signals
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Calculate median price
        median = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_hl = highest - lowest
        if range_hl < 1e-10:
            fisher[i] = 0.0
            trigger[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to 0-1 range
        normalized = (median - lowest) / range_hl
        
        # Clamp to avoid division by zero
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher calculation
        fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value (Ehlers smoothing)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
        else:
            fisher[i] = fisher_val
        
        # Trigger is previous Fisher value
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, trigger

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """
    ATR Ratio for volatility spike detection
    Ratio > 1.8 = vol spike (panic/euphoria) = mean reversion opportunity
    """
    n = len(close)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    # Calculate true range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate ATRs
    atr_short = pd.Series(tr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(tr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    # Ratio
    ratio = np.zeros(n)
    ratio[:] = np.nan
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_HIGH = 0.30  # 30% for high-conviction (vol spike + Fisher + HTF)
    SIZE_NORMAL = 0.20  # 20% for normal entries
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crosses to avoid repeated signals
    last_fisher_cross = 0  # 0=none, 1=bull, -1=bear
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_ratio[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h + 1d HMA) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both HTFs agree
        htf_strong_bull = htf_12h_bull and htf_1d_bull
        htf_strong_bear = htf_12h_bear and htf_1d_bear
        
        # === VOLATILITY REGIME ===
        vol_spike = atr_ratio[i] > 1.8  # Panic/euphoria = mean reversion opportunity
        vol_normal = atr_ratio[i] <= 1.8
        
        # === FISHER TRANSFORM SIGNALS ===
        # Bullish cross: Fisher crosses above -1.5 (oversold reversal)
        fisher_bull_cross = (fisher_trigger[i] < -1.5) and (fisher[i] >= -1.5)
        # Bearish cross: Fisher crosses below +1.5 (overbought reversal)
        fisher_bear_cross = (fisher_trigger[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher extreme levels (for vol spike entries)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = SIZE_NORMAL
        
        # HIGH-CONVICTION LONG: Vol spike + Fisher oversold + HTF not bear
        if vol_spike and fisher_oversold and not htf_strong_bear and rsi_oversold:
            desired_signal = SIZE_HIGH
            signal_strength = SIZE_HIGH
            last_fisher_cross = 1
        # HIGH-CONVICTION SHORT: Vol spike + Fisher overbought + HTF not bull
        elif vol_spike and fisher_overbought and not htf_strong_bull and rsi_overbought:
            desired_signal = -SIZE_HIGH
            signal_strength = SIZE_HIGH
            last_fisher_cross = -1
        # NORMAL LONG: Fisher bull cross + HTF bull bias + HMA bull
        elif fisher_bull_cross and htf_12h_bull and hma_bull and last_fisher_cross != 1:
            desired_signal = SIZE_NORMAL
            last_fisher_cross = 1
        # NORMAL SHORT: Fisher bear cross + HTF bear bias + HMA bear
        elif fisher_bear_cross and htf_12h_bear and hma_bear and last_fisher_cross != -1:
            desired_signal = -SIZE_NORMAL
            last_fisher_cross = -1
        # TREND FOLLOW LONG: Strong HTF bull + pullback to HMA + RSI ok
        elif htf_strong_bull and hma_bull and rsi[i] > 40.0 and rsi[i] < 70.0:
            desired_signal = SIZE_NORMAL * 0.7
        # TREND FOLLOW SHORT: Strong HTF bear + rally to HMA + RSI ok
        elif htf_strong_bear and hma_bear and rsi[i] > 30.0 and rsi[i] < 60.0:
            desired_signal = -SIZE_NORMAL * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
            last_fisher_cross = 0  # Reset on stoploss
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_HIGH * 0.85:
            final_signal = SIZE_HIGH
        elif desired_signal <= -SIZE_HIGH * 0.85:
            final_signal = -SIZE_HIGH
        elif desired_signal >= SIZE_NORMAL * 0.85:
            final_signal = SIZE_NORMAL
        elif desired_signal <= -SIZE_NORMAL * 0.85:
            final_signal = -SIZE_NORMAL
        elif abs(desired_signal) >= SIZE_NORMAL * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_NORMAL * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                last_fisher_cross = position_side
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals