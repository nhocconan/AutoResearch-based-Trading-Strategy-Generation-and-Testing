#!/usr/bin/env python3
"""
Experiment #1401: 4h Primary + 1d/1w HTF — Dual HMA Trend + Donchian Breakout

Hypothesis: 4h strategies failed due to over-complicated regime filters (Choppiness + CRSI)
that over-filtered signals. The working patterns (#1391 4h Sharpe=0.305, #1396 12h Sharpe=0.525)
used SIMPLE trend following: HTF HMA + Donchian breakout + RSI + ATR stop.

Key insight: Use DUAL HTF confirmation (1w for macro bias, 1d for intermediate trend) to
improve signal quality across ALL symbols (BTC/ETH/SOL). 1w HMA filters out counter-trend
trades during major bear markets (2022 crash). 1d HMA provides intermediate confirmation.
4h Donchian(20) breakout gives entry timing with adequate frequency.

Design:
1. 1w HMA(21) = ultra-long-term macro bias (only trade with 1w trend)
2. 1d HMA(21) = intermediate trend confirmation (strengthens signal when aligned with 1w)
3. 4h Donchian(20) breakout = entry trigger (proven on 12h/1d)
4. RSI(14) momentum filter (wide bands 30-70 to ensure >=30 trades/train)
5. ATR(14) trailing stop 2.5x = risk management
6. Position size: 0.30 when 1w+1d aligned, 0.20 when only 1d aligned
7. NO regime filter (Choppiness/CRSI failed on 4h in #1394, #1397)

Target: 20-50 trades/year, Sharpe > 0.618 (beat 1d baseline), trades >= 30 train, >= 5 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_1d1w_rsi_atr_dual_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - wide bands for entry confirmation"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels for entry trigger"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for ultra-long-term macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for intermediate trend confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE_STRONG = 0.30  # When 1w + 1d aligned
    BASE_SIZE_WEAK = 0.20    # When only 1d aligned
    
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
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
        
        # === MACRO TREND (1w HMA) - ultra-long-term bias ===
        macro_1w_bull = close[i] > hma_1w_aligned[i]
        macro_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) - confirmation ===
        macro_1d_bull = close[i] > hma_1d_aligned[i]
        macro_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === HTF CONFLUENCE ===
        htf_aligned_bull = macro_1w_bull and macro_1d_bull
        htf_aligned_bear = macro_1w_bear and macro_1d_bear
        htf_mixed = (macro_1w_bull and macro_1d_bear) or (macro_1w_bear and macro_1d_bull)
        
        # === RSI MOMENTUM (WIDE bands to ensure trades) ===
        rsi_bull = rsi[i] > 30.0
        rsi_bear = rsi[i] < 70.0
        rsi_neutral = (rsi[i] > 35.0) and (rsi[i] < 65.0)
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === DESIRED SIGNAL - DUAL HTF CONFIRMATION ===
        desired_signal = 0.0
        position_size = 0.0
        
        # LONG ENTRY PATHS
        # Path 1: Breakout + 1w+1d aligned (strongest signal)
        if breakout_long and htf_aligned_bull and rsi_bull:
            desired_signal = 1.0
            position_size = BASE_SIZE_STRONG
        # Path 2: Breakout + 1d aligned only (moderate signal)
        elif breakout_long and macro_1d_bull and rsi_bull and not htf_mixed:
            desired_signal = 1.0
            position_size = BASE_SIZE_WEAK
        # Path 3: 1w+1d aligned + RSI momentum (trend continuation without breakout)
        elif htf_aligned_bull and rsi[i] > 45.0 and not breakout_short:
            desired_signal = 0.5
            position_size = BASE_SIZE_WEAK
        
        # SHORT ENTRY PATHS
        # Path 1: Breakout + 1w+1d aligned (strongest signal)
        elif breakout_short and htf_aligned_bear and rsi_bear:
            desired_signal = -1.0
            position_size = BASE_SIZE_STRONG
        # Path 2: Breakout + 1d aligned only (moderate signal)
        elif breakout_short and macro_1d_bear and rsi_bear and not htf_mixed:
            desired_signal = -1.0
            position_size = BASE_SIZE_WEAK
        # Path 3: 1w+1d aligned + RSI momentum (trend continuation without breakout)
        elif htf_aligned_bear and rsi[i] < 55.0 and not breakout_long:
            desired_signal = -0.5
            position_size = BASE_SIZE_WEAK
        
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
            position_size = 0.0
        
        # === FINAL SIGNAL ===
        if desired_signal != 0.0 and position_size > 0:
            final_signal = position_size if desired_signal > 0 else -position_size
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