#!/usr/bin/env python3
"""
Experiment #938: 4h Primary + 1d HTF — KAMA Adaptive Trend + RSI Pullback + Choppiness Regime

Hypothesis: Kaufman Adaptive Moving Average (KAMA) outperforms HMA/EMA in mixed 
regime markets (2021 bull + 2022 crash + 2023-24 range + 2025 bear). KAMA adapts 
to volatility - fast in trends, slow in chop. Combined with 1d HMA bias and 
Choppiness Index regime filter, this should handle both trending and ranging 
periods better than pure trend-following.

Key innovations:
1. KAMA(10,2,30) adapts efficiency ratio - faster in trends, slower in chop
2. 1d HMA(21) for HTF bias - proven directional filter
3. Choppiness Index(14) regime switch - CHOP>61.8 = mean revert, CHOP<38.2 = trend
4. RSI(14) pullback entries - enter on dips in uptrend, rallies in downtrend
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
7. LOOSE entry conditions to guarantee ≥10 trades/train, ≥3/test

Entry logic (LOOSE to ensure trades):
- TREND MODE (CHOP < 50): KAMA crossover + 1d HMA bias + RSI not extreme
- RANGE MODE (CHOP >= 50): RSI extremes (30/70) + 1d HMA bias
- This dual-mode approach captures both 2021 trend and 2025 range markets

Target: Sharpe>0.45, trades>=20 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_chop_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    Smoothing Constant (SC) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA[prev] + SC * (Close - KAMA[prev])
    """
    n = len(close)
    if n < slow_period + 1:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Initialize KAMA at first valid point
    kama[slow_period] = close[slow_period]
    
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    for i in range(slow_period + 1, n):
        # Calculate Efficiency Ratio
        signal = abs(close[i] - close[i - er_period])
        
        noise = 0.0
        for j in range(1, er_period + 1):
            noise += abs(close[i - j + 1] - close[i - j])
        
        if noise > 1e-10:
            er = signal / noise
        else:
            er = 0.0
        
        # Calculate Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Update KAMA
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = choppy/ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Also calculate KAMA for crossover signal (dual KAMA)
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    kama_slow = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trending = chop_14[i] < 50.0  # Midpoint between 38.2 and 61.8
        is_choppy = chop_14[i] >= 50.0
        
        # === KAMA CROSSOVER ===
        kama_cross_long = False
        kama_cross_short = False
        if i > 0 and not np.isnan(kama_fast[i-1]) and not np.isnan(kama_slow[i-1]):
            kama_cross_long = (kama_fast[i-1] <= kama_slow[i-1]) and (kama_fast[i] > kama_slow[i])
            kama_cross_short = (kama_fast[i-1] >= kama_slow[i-1]) and (kama_fast[i] < kama_slow[i])
        
        # === KAMA TREND ===
        kama_bull = kama_fast[i] > kama_slow[i]
        kama_bear = kama_fast[i] < kama_slow[i]
        
        # === RSI LEVELS ===
        rsi_overbought = rsi_14[i] > 70.0
        rsi_oversold = rsi_14[i] < 30.0
        rsi_bullish = rsi_14[i] > 50.0
        rsi_bearish = rsi_14[i] < 50.0
        
        # === ENTRY LOGIC (DUAL REGIME - LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND MODE: Follow KAMA direction with HTF bias
            if htf_1d_bull:
                # Long in uptrend
                if kama_cross_long:
                    desired_signal = SIZE_STRONG
                elif kama_bull and rsi_bullish and not rsi_overbought:
                    desired_signal = SIZE_BASE
            elif htf_1d_bear:
                # Short in downtrend
                if kama_cross_short:
                    desired_signal = -SIZE_STRONG
                elif kama_bear and rsi_bearish and not rsi_oversold:
                    desired_signal = -SIZE_BASE
        else:
            # CHOPPY MODE: Mean reversion at RSI extremes with HTF bias
            if htf_1d_bull:
                # Long on RSI pullback in bullish HTF
                if rsi_oversold or (rsi_14[i] < 40.0 and rsi_14[i-1] >= 40.0):
                    desired_signal = SIZE_BASE
            elif htf_1d_bear:
                # Short on RSI rally in bearish HTF
                if rsi_overbought or (rsi_14[i] > 60.0 and rsi_14[i-1] <= 60.0):
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