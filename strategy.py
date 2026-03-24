#!/usr/bin/env python3
"""
Experiment #842: 4h Primary + 1d/1w HTF — Adaptive KAMA with Loose RSI Entries

Hypothesis: 4h timeframe with dual HTF bias (1w + 1d) provides strong trend filtering
while KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA in choppy
markets. Key innovation: LOOSE entry thresholds (RSI 40/60 not 30/70) to GUARANTEE
trade generation while HTF filters maintain edge.

Why this should beat Sharpe=0.424 baseline:
1. KAMA adapts to volatility — reduces whipsaw in range markets (2022, 2025)
2. Dual HTF (1w + 1d) — stronger trend bias than single HTF
3. Choppiness Index adjusts SIZE not blocks entries — ensures trades still happen
4. RSI thresholds 40/60 — much looser than typical 30/70, guarantees ≥30 trades
5. ATR 2.5x trailing stop — proven risk management

Entry conditions (LOOSE for trade generation):
- LONG: 1w HMA bull + 1d HMA bull + (RSI<50 OR KAMA bull)
- SHORT: 1w HMA bear + 1d HMA bear + (RSI>50 OR KAMA bear)

Target: Sharpe>0.50, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 4h (MANDATORY per experiment)
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_dual_htf_loose_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market volatility"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        sum_volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if sum_volatility > 1e-10:
            er[i] = price_change / sum_volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies ranging vs trending markets"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum([max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1])) 
                          for j in range(i-period+1, i+1)])
        
        if highest_high - lowest_low > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / atr_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_fast = calculate_kama(close, period=5, fast_period=2, slow_period=15)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_4h[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
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
        
        # === HTF BIAS (1w + 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Dual HTF confirmation (both must agree for strong signal)
        htf_strong_bull = htf_1w_bull and htf_1d_bull
        htf_strong_bear = htf_1w_bear and htf_1d_bear
        htf_weak_bull = htf_1w_bull or htf_1d_bull
        htf_weak_bear = htf_1w_bear or htf_1d_bear
        
        # === 4h KAMA TREND ===
        kama_bull = close[i] > kama_4h[i]
        kama_bear = close[i] < kama_4h[i]
        kama_crossover_long = False
        kama_crossover_short = False
        if i > 0 and not np.isnan(kama_fast[i-1]) and not np.isnan(kama_4h[i-1]):
            kama_crossover_long = (kama_fast[i-1] <= kama_4h[i-1]) and (kama_fast[i] > kama_4h[i])
            kama_crossover_short = (kama_fast[i-1] >= kama_4h[i-1]) and (kama_fast[i] < kama_4h[i])
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        rsi_oversold = rsi_14[i] < 50.0  # Very loose - was 30, now 50
        rsi_overbought = rsi_14[i] > 50.0  # Very loose - was 70, now 50
        rsi_extreme_oversold = rsi_14[i] < 35.0
        rsi_extreme_overbought = rsi_14[i] > 65.0
        
        # === CHOPPINESS REGIME (adjust size, don't block) ===
        chop_range = chop_14[i] > 50.0  # Ranging market
        chop_trend = chop_14[i] < 50.0  # Trending market
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        signal_strength = 1.0
        
        # LONG: HTF bull + (RSI<50 OR KAMA bull OR crossover)
        if htf_strong_bull:
            if rsi_oversold or kama_bull or kama_crossover_long:
                if rsi_extreme_oversold or kama_crossover_long:
                    desired_signal = SIZE_STRONG
                elif chop_trend:
                    desired_signal = SIZE_BASE
                else:
                    desired_signal = SIZE_WEAK
        elif htf_weak_bull:
            if rsi_extreme_oversold or kama_crossover_long:
                desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + (RSI>50 OR KAMA bear OR crossover)
        elif htf_strong_bear:
            if rsi_overbought or kama_bear or kama_crossover_short:
                if rsi_extreme_overbought or kama_crossover_short:
                    desired_signal = -SIZE_STRONG
                elif chop_trend:
                    desired_signal = -SIZE_BASE
                else:
                    desired_signal = -SIZE_WEAK
        elif htf_weak_bear:
            if rsi_extreme_overbought or kama_crossover_short:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals