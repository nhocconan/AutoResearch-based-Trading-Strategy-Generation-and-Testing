#!/usr/bin/env python3
"""
Experiment #1249: 4h Primary + 1d HTF — Adaptive KAMA Dual Regime Strategy

Hypothesis: KAMA adapts to market noise better than HMA/EMA. Combined with 
Choppiness Index regime detection, we can switch between trend-following 
(in low chop) and mean-reversion (in high chop). This addresses the key 
failure mode of pure trend strategies in 2022 crash and 2025 bear market.

Key innovations vs #1239:
1. KAMA instead of HMA - adapts speed based on Efficiency Ratio
2. Choppiness Index regime filter - trend vs mean-revert mode
3. Dual entry logic: trend breakout OR BB mean-reversion based on regime
4. Relaxed RSI thresholds (35-65 vs 40-60) for more trade frequency
5. Lower ADX threshold (15 vs 18) to increase signals

Target: Sharpe > 0.612, trades >= 80 train (20/year), >= 12 test (3/year)
Timeframe: 4h (20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_regime_1d_dual_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures if market is trending or ranging"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        
        if sum_tr > 1e-10 and (highest_high - lowest_low) > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return middle, upper, lower
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return middle, upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for macro trend filter
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=20)
    kama_slow = calculate_kama(close, er_period=10, fast_period=2, slow_period=40)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Slightly lower for safety
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Hysteresis: require signal change to persist for 2 bars
    signal_buffer = 0
    last_signal = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === MACRO TREND (1d KAMA) ===
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
        is_trending = chop[i] < 45.0  # Slightly relaxed threshold
        is_ranging = chop[i] > 55.0
        
        # === KAMA CROSSOVER (Trend Following) ===
        kama_bull = kama_fast[i] > kama_slow[i]
        kama_bear = kama_fast[i] < kama_slow[i]
        
        # === RSI FILTER (Relaxed for more trades) ===
        rsi_neutral_long = rsi[i] > 35.0 and rsi[i] < 65.0
        rsi_neutral_short = rsi[i] > 35.0 and rsi[i] < 65.0
        
        # === BOLLINGER MEAN REVERSION ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow KAMA crossover with macro alignment
        if is_trending:
            # LONG: Macro bull + KAMA bull crossover + RSI not extreme
            if macro_bull and kama_bull and rsi_neutral_long:
                desired_signal = BASE_SIZE
            # SHORT: Macro bear + KAMA bear crossover + RSI not extreme
            elif macro_bear and kama_bear and rsi_neutral_short:
                desired_signal = -BASE_SIZE
        
        # RANGE REGIME: Mean revert at Bollinger extremes
        elif is_ranging:
            # LONG: Price below lower BB + macro not strongly bear
            if bb_oversold and not macro_bear:
                desired_signal = BASE_SIZE
            # SHORT: Price above upper BB + macro not strongly bull
            elif bb_overbought and not macro_bull:
                desired_signal = -BASE_SIZE
        
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
        
        # === HYSTERESIS (prevent rapid flipping) ===
        if desired_signal != last_signal:
            signal_buffer += 1
            if signal_buffer >= 2:
                last_signal = desired_signal
                signal_buffer = 0
        else:
            signal_buffer = 0
        
        final_signal = last_signal
        
        # === DISCRETIZE SIGNAL VALUES ===
        if final_signal > 0:
            final_signal = BASE_SIZE
        elif final_signal < 0:
            final_signal = -BASE_SIZE
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