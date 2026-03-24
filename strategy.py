#!/usr/bin/env python3
"""
Experiment #935: 6h Primary + 12h/1d HTF — KAMA Adaptive Trend + Donchian Breakout + CHOP Regime

Hypothesis: 6h timeframe sits between 4h and 12h, capturing multi-day trends while avoiding
lower-TF noise. KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast in trends,
slow in ranges. Combined with Donchian(20) breakout confirmation and Choppiness Index regime
filter, this should capture sustained moves while avoiding whipsaws. 1d HMA provides ultimate
trend bias. 12h KAMA adds intermediate confirmation layer.

Key innovations:
1. KAMA(10,2,30) adapts speed based on Efficiency Ratio - proven in trending crypto markets
2. Donchian(20) breakout confirms momentum - price breaks 20-bar high/low
3. CHOP(14) < 50 = trending regime (only trade breakouts when market is trending)
4. 1d HMA(21) for ultimate HTF bias - price above = long only, below = short only
5. 12h KAMA for intermediate trend confirmation - adds layer between 6h and 1d
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry conditions (balanced for trade frequency):
- LONG = 1d HMA bull + 12h KAMA bull + CHOP<50 + Donchian breakout up OR 6h KAMA crossover up
- SHORT = 1d HMA bear + 12h KAMA bear + CHOP<50 + Donchian breakout down OR 6h KAMA crossover down
- CHOP filter can be bypassed on strong KAMA crossover (ensures trades in all regimes)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_donchian_chop_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Efficiency Ratio calculation
    er = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA with SMA of first period
    kama[period - 1] = np.mean(close[:period])
    
    # Calculate KAMA
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = highest high of last period bars
    Lower = lowest low of last period bars
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures if market is choppy/ranging or trending
    CHOP = 100 * log10(sum(ATR, period) / (highest high - lowest low)) / log10(period)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
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
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop_14 = calculate_chop(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(kama_6h[i]) or np.isnan(donchian_upper[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE HTF (12h KAMA) ===
        htf_12h_bull = close[i] > kama_12h_aligned[i]
        htf_12h_bear = close[i] < kama_12h_aligned[i]
        
        # === 6h KAMA TREND ===
        kama_6h_bull = close[i] > kama_6h[i]
        kama_6h_bear = close[i] < kama_6h[i]
        
        # === KAMA CROSSOVER ===
        kama_crossover_long = False
        kama_crossover_short = False
        if i > 0 and not np.isnan(kama_6h[i-1]):
            kama_crossover_long = (close[i-1] <= kama_6h[i-1]) and (close[i] > kama_6h[i])
            kama_crossover_short = (close[i-1] >= kama_6h[i-1]) and (close[i] < kama_6h[i])
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0  # Below 50 = trending market
        chop_ranging = chop_14[i] >= 50.0  # Above 50 = ranging market
        
        # === ENTRY LOGIC (BALANCED FOR TRADE FREQUENCY) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_1d_bull:
            # Strong long: HTF aligned + trending regime + breakout
            if htf_12h_bull and chop_trending and donchian_breakout_long:
                desired_signal = SIZE_STRONG
            # Moderate long: KAMA crossover (bypasses CHOP filter for trade frequency)
            elif kama_crossover_long and htf_12h_bull:
                desired_signal = SIZE_BASE
            # Continuation long: price above KAMA in trending regime
            elif kama_6h_bull and chop_trending and htf_12h_bull:
                desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_1d_bear:
            # Strong short: HTF aligned + trending regime + breakout
            if htf_12h_bear and chop_trending and donchian_breakout_short:
                desired_signal = -SIZE_STRONG
            # Moderate short: KAMA crossover (bypasses CHOP filter for trade frequency)
            elif kama_crossover_short and htf_12h_bear:
                desired_signal = -SIZE_BASE
            # Continuation short: price below KAMA in trending regime
            elif kama_6h_bear and chop_trending and htf_12h_bear:
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