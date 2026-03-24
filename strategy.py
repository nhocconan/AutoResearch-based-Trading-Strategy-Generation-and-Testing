#!/usr/bin/env python3
"""
Experiment #034: 4h Primary + 12h/1d HTF — Dual Regime (Choppiness + HMA + BB)

Hypothesis: After 33 experiments, the clearest pattern is that SINGLE-REGIME strategies
fail in mixed markets. BTC/ETH 2021-2024 includes bull (2021), crash (2022), recovery
(2023-2024). A strategy that adapts to market regime should outperform.

This strategy uses Choppiness Index (CHOP) to detect regime:
- CHOP > 61.8 = RANGING → Mean reversion (Bollinger Band edges)
- CHOP < 38.2 = TRENDING → Trend following (HMA + RSI pullback)
- 38.2 <= CHOP <= 61.8 = TRANSITION → Stay flat or reduce size

Key innovations vs failed experiments:
1. CHOP(14) regime detection (proven on ETH with Sharpe +0.923 in research)
2. 12h HMA for HTF bias (not 1d - too slow for 4h entries)
3. Dual entry logic per regime (not one-size-fits-all)
4. LOOSE thresholds: RSI 35/65 (not 30/70), CHOP 55/65 (not 61.8 exact)
5. Size reduction in transition regime (0.15 vs 0.30) to reduce whipsaw

Entry Logic:
- TREND regime (CHOP<45): Long if 12h_HMA_bull + 4h_HMA_bull + RSI<55
- TREND regime (CHOP<45): Short if 12h_HMA_bear + 4h_HMA_bear + RSI>45
- RANGE regime (CHOP>60): Long if price<BB_lower + RSI<40
- RANGE regime (CHOP>60): Short if price>BB_upper + RSI>60
- TRANSITION: Size=0.15 or flat

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.22 (beat current best), trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 4h (target 25-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_dual_regime_12h_v2"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    double_wma_half = 2.0 * wma_half - wma_full
    hma = wma(double_wma_half, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum filter"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
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
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = Ranging market (mean reversion works)
    CHOP < 38.2 = Trending market (trend following works)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10 or atr_sum < 1e-10:
            chop[i] = 50.0  # neutral
        else:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands - for mean reversion entries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for HTF trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=9)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # Full size in clear trend/range
    SIZE_TRANSITION = 0.15  # Half size in transition
    SIZE_RANGE = 0.25  # Slightly smaller for mean reversion
    
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
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 60 = Range, CHOP < 45 = Trend, else = Transition
        is_trend_regime = chop[i] < 45.0
        is_range_regime = chop[i] > 60.0
        is_transition = not is_trend_regime and not is_range_regime
        
        # === HTF BIAS (12h HMA) ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === 4h TREND ===
        hma_4h_bull = close[i] > hma_4h[i]
        hma_4h_bear = close[i] < hma_4h[i]
        hma_fast_above_slow = hma_4h_fast[i] > hma_4h[i] if not np.isnan(hma_4h_fast[i]) else False
        hma_fast_below_slow = hma_4h_fast[i] < hma_4h[i] if not np.isnan(hma_4h_fast[i]) else False
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        current_size = SIZE_TREND
        
        if is_trend_regime:
            # TREND FOLLOWING: Trade with HTF + 4h trend, enter on RSI pullback
            current_size = SIZE_TREND
            
            # LONG: 12h bull + 4h bull + fast>slow + RSI pullback < 55
            if hma_12h_bull and hma_4h_bull and hma_fast_above_slow and rsi[i] < 55.0:
                desired_signal = current_size
            # SHORT: 12h bear + 4h bear + fast<slow + RSI rally > 45
            elif hma_12h_bear and hma_4h_bear and hma_fast_below_slow and rsi[i] > 45.0:
                desired_signal = -current_size
        
        elif is_range_regime:
            # MEAN REVERSION: Fade Bollinger Band extremes with RSI confirmation
            current_size = SIZE_RANGE
            
            # LONG: Price below BB lower + RSI oversold < 40
            if close[i] < bb_lower[i] and rsi[i] < 40.0:
                desired_signal = current_size
            # SHORT: Price above BB upper + RSI overbought > 60
            elif close[i] > bb_upper[i] and rsi[i] > 60.0:
                desired_signal = -current_size
        
        else:
            # TRANSITION REGIME: Reduce size or stay flat
            current_size = SIZE_TRANSITION
            
            # Only trade if strong confluence
            if hma_12h_bull and hma_4h_bull and rsi[i] < 45.0:
                desired_signal = current_size
            elif hma_12h_bear and hma_4h_bear and rsi[i] > 55.0:
                desired_signal = -current_size
        
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
        if desired_signal >= current_size * 0.85:
            final_signal = current_size
        elif desired_signal <= -current_size * 0.85:
            final_signal = -current_size
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