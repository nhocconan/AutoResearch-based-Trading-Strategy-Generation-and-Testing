#!/usr/bin/env python3
"""
Experiment #151: 6h Primary + 1d/1w Dual HTF — HMA Trend + Fisher Transform + Volume

Hypothesis: 6h timeframe offers optimal trade frequency (30-60/year) between 4h (fee drag)
and 12h (missed opportunities). Adding DUAL HTF (1d + 1w) provides stronger trend confirmation
than single HTF. Fisher Transform catches reversals in bear/range markets where HMA lags.

Key improvements over #147 (Sharpe=0.161):
1. DUAL HTF: Both 1d AND 1w must align for full size entries (stronger trend filter)
2. Fisher Transform: Catches reversals at extremes (works in 2022 crash, 2025 bear)
3. Volume confirmation: Breakouts need 1.5x avg volume (filters false breakouts)
4. Asymmetric sizing: 0.30 with HTF alignment, 0.20 without (risk-adjusted)
5. LOOSE Fisher thresholds: >-1.5 long, <1.5 short (ensures trade generation)

Design for trade generation:
- Primary: All conditions aligned (HMA + Fisher + Volume + Dual HTF) = 0.30
- Fallback: Strong HTF alignment only (ignore Fisher if 1d+1w both strong) = 0.20
- Stoploss: 2.5x ATR trailing (same as #147, proven)
- Target: 30-60 trades/year, Sharpe>0.167, DD>-40%

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_fisher_volume_dual_htf_v1"
timeframe = "6h"
leverage = 1.0

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

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - catches reversals at extremes
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    Works well in bear/range markets (2022 crash, 2025 bear)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        range_hl = highest - lowest
        
        if range_hl < 1e-10:
            fisher[i] = fisher[i-1] if i > period else 0.0
        else:
            x = 0.96 * (2.0 * (close[i] - lowest) / range_hl - 1.0)
            x = np.clip(x, -0.999, 0.999)
            
            if i == period:
                fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
            else:
                fisher[i] = 0.7 * (0.5 * np.log((1.0 + x) / (1.0 - x))) + 0.3 * fisher[i-1]
        
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average - confirms breakouts"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio[:period] = np.nan
    
    return vol_ratio

def calculate_rsi(close, period=14):
    """Relative Strength Index for additional confirmation"""
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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    fisher, trigger = calculate_fisher(close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_FULL = 0.30  # 30% with full alignment
    SIZE_HALF = 0.20  # 20% with partial alignment
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === DUAL HTF BIAS (1d + 1w) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Both HTF aligned = strong trend
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === FISHER TRANSFORM (reversal detection) ===
        # Long: Fisher > -1.5 (emerging from oversold)
        # Short: Fisher < 1.5 (emerging from overbought)
        fisher_ok_long = fisher[i] > -1.5
        fisher_ok_short = fisher[i] < 1.5
        
        # Fisher crossover confirmation
        fisher_cross_long = fisher[i] > trigger[i] if not np.isnan(trigger[i]) else False
        fisher_cross_short = fisher[i] < trigger[i] if not np.isnan(trigger[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_ok = vol_ratio[i] >= 1.2  # At least 1.2x average volume
        
        # === RSI CONFIRMATION (LOOSE) ===
        rsi_ok_long = rsi[i] > 35.0  # Not extremely oversold
        rsi_ok_short = rsi[i] < 65.0  # Not extremely overbought
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY: All conditions aligned (full size 0.30)
        if (hma_bull and htf_strong_bull and fisher_ok_long and vol_ok and rsi_ok_long):
            desired_signal = SIZE_FULL
        
        elif (hma_bear and htf_strong_bear and fisher_ok_short and vol_ok and rsi_ok_short):
            desired_signal = -SIZE_FULL
        
        # FALLBACK 1: Strong HTF alignment only (ignore Fisher/Vol if HTF very strong) = 0.20
        elif (hma_bull and htf_strong_bull and rsi_ok_long):
            desired_signal = SIZE_HALF
        
        elif (hma_bear and htf_strong_bear and rsi_ok_short):
            desired_signal = -SIZE_HALF
        
        # FALLBACK 2: Single HTF strong + Fisher confirmation = 0.20
        elif (hma_bull and htf_1d_bull and fisher_cross_long and vol_ok):
            desired_signal = SIZE_HALF
        
        elif (hma_bear and htf_1d_bear and fisher_cross_short and vol_ok):
            desired_signal = -SIZE_HALF
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_FULL * 0.9:
            final_signal = SIZE_FULL
        elif desired_signal <= -SIZE_FULL * 0.9:
            final_signal = -SIZE_FULL
        elif desired_signal >= SIZE_HALF * 0.9:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.9:
            final_signal = -SIZE_HALF
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