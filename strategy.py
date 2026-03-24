#!/usr/bin/env python3
"""
Experiment #029: 4h Primary + 12h HTF — Fisher Transform Dual Regime with ADX Filter

Hypothesis: Building on #019's success (Sharpe=0.368), this replaces RSI with Fisher Transform
which is more sensitive to price extremes and catches reversals faster in bear/range markets.
The 12h HTF provides more responsive trend filtering than 1d while avoiding 4h noise.

Key changes from #019:
1. Fisher Transform (period=9) instead of RSI - superior reversal detection in crypto
2. 12h HTF instead of 1d - more trade opportunities, faster regime adaptation
3. ADX + Choppiness dual regime filter - ADX>25 = trend, CHOP>55 = range
4. Asymmetric entry thresholds based on regime state
5. ATR-based dynamic position sizing with volatility scaling

Entry Logic:
- RANGE (CHOP>55): Fisher<-1.5 long, Fisher>1.5 short (mean reversion)
- TREND (ADX>25 + CHOP<45): 12h HMA bias + Fisher confirmation
- NEUTRAL: Only trade with strong HTF alignment
- Size: 0.30 with HTF alignment, 0.25 without

Risk: 2.5x ATR trailing stop, max signal 0.35, discrete levels (0.0, ±0.25, ±0.30)
Target: Sharpe>0.4, trades>40/symbol train, >4/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_adx_chop_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Transforms price into a Gaussian normal distribution
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    # Normalize price to -1 to +1 range
    for i in range(period - 1, n):
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize: (close - lowest) / (highest - lowest)
        normalized = 2.0 * (close[i] - lowest) / (highest - lowest) - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform: 0.5 * ln((1 + normalized) / (1 - normalized))
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Signal line (1-period lag)
        if i > 0 and not np.isnan(fisher[i - 1]):
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength regardless of direction
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    adx = np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth TR, +DM, -DM using Wilder's smoothing
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    plus_di[period-1] = 100.0 * np.mean(plus_dm[:period]) / atr[period-1] if atr[period-1] > 1e-10 else 0.0
    minus_di[period-1] = 100.0 * np.mean(minus_dm[:period]) / atr[period-1] if atr[period-1] > 1e-10 else 0.0
    
    for i in range(period, n):
        plus_di[i] = 100.0 * ((plus_di[i-1] * (period - 1) + 100.0 * plus_dm[i] / atr[i]) / period) if atr[i] > 1e-10 else 0.0
        minus_di[i] = 100.0 * ((minus_di[i-1] * (period - 1) + 100.0 * minus_dm[i] / atr[i]) / period) if atr[i] > 1e-10 else 0.0
    
    # DX and ADX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # ADX = smoothed DX
    adx[period * 2 - 1] = np.mean(dx[period:period*2])
    for i in range(period * 2, n):
        if np.isnan(dx[i]):
            continue
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_hma(close, period=21):
    """Hull Moving Average - smooth and responsive"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_signal = calculate_fisher(close, period=9)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.25
    MAX_SIZE = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crosses to avoid repeated signals
    prev_fisher = 0.0
    prev_fisher_valid = False
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(adx[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 55.0
        is_trending = adx[i] > 25.0 and chop[i] < 45.0
        is_neutral = not is_choppy and not is_trending
        
        # === HTF TREND BIAS (12h) ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # Fisher cross detection (only trigger on cross, not level)
        fisher_cross_long = False
        fisher_cross_short = False
        
        if prev_fisher_valid:
            # Long: Fisher crosses above -1.5 from below
            if prev_fisher < -1.5 and fisher[i] >= -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 from above
            if prev_fisher > 1.5 and fisher[i] <= 1.5:
                fisher_cross_short = True
        
        if is_choppy:
            # MEAN REVERSION REGIME - trade Fisher extremes
            if fisher_cross_long:
                signal_strength = BASE_SIZE if hma_12h_bull else REDUCED_SIZE
                desired_signal = signal_strength
            elif fisher_cross_short:
                signal_strength = BASE_SIZE if hma_12h_bear else REDUCED_SIZE
                desired_signal = -signal_strength
            # Also allow entries at extreme levels without cross (for more trades)
            elif fisher[i] < -1.8:
                signal_strength = REDUCED_SIZE if hma_12h_bull else BASE_SIZE * 0.5
                desired_signal = signal_strength
            elif fisher[i] > 1.8:
                signal_strength = REDUCED_SIZE if hma_12h_bear else BASE_SIZE * 0.5
                desired_signal = -signal_strength
        
        elif is_trending:
            # TREND REGIME - trade with HTF bias only
            if hma_12h_bull and fisher_cross_long:
                desired_signal = BASE_SIZE
            elif hma_12h_bull and fisher[i] < -1.0:
                # Pullback entry in uptrend
                desired_signal = REDUCED_SIZE
            elif hma_12h_bear and fisher_cross_short:
                desired_signal = -BASE_SIZE
            elif hma_12h_bear and fisher[i] > 1.0:
                # Pullback entry in downtrend
                desired_signal = -REDUCED_SIZE
        
        else:
            # NEUTRAL REGIME - only trade with strong HTF alignment + Fisher confirmation
            if hma_12h_bull and fisher[i] < -1.0:
                desired_signal = REDUCED_SIZE
            elif hma_12h_bear and fisher[i] > 1.0:
                desired_signal = -REDUCED_SIZE
        
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
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
        elif abs(desired_signal) >= 0.10:
            final_signal = np.sign(desired_signal) * REDUCED_SIZE
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
        
        # Update Fisher tracking
        prev_fisher = fisher[i]
        prev_fisher_valid = True
    
    return signals