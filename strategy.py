#!/usr/bin/env python3
"""
Experiment #1382: 12h Primary + 1d/1w HTF — Vol Spike Reversion + Asymmetric Regime

Hypothesis: Previous 12h strategies (#1372, #1376) failed because they used symmetric
trend-following logic that gets whipsawed in bear/range markets (2022 crash, 2025 bear).
Research shows BTC/ETH work better with: (1) vol spike reversion, (2) asymmetric regime
(only short in bear, only long in bull), (3) BB squeeze before breakout.

Key insight from market analysis: Simple trend following ALWAYS fails on BTC/ETH during
crashes. Need regime-adaptive logic that switches between mean-reversion (chop) and
trend-following (breakout) based on volatility compression.

Design:
1. 1w HMA(21) = ultra-macro trend bias (soft filter)
2. 1d HMA(21) = regime filter (bull: price>HMA, bear: price<HMA)
3. 12h BB Width percentile = squeeze detection (vol compression before expansion)
4. 12h ATR ratio(7/30) = vol spike timing (entry trigger)
5. 12h RSI(14) asymmetric thresholds = momentum confirmation
6. ATR(14) trailing stop 2.5x = risk management
7. Position size 0.30 = conservative for 12h
8. Asymmetric entries: long only in bull regime, short only in bear regime

Target: 25-45 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vol_spike_asymmetric_regime_bb_squeeze_1d1w_atr_v1"
timeframe = "12h"
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
    """Relative Strength Index"""
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with width for squeeze detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma * 100.0
    
    return upper, lower, width

def calculate_bb_width_percentile(bb_width, lookback=60):
    """BB Width percentile over lookback period - detects squeeze"""
    n = len(bb_width)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        window = bb_width[i - lookback + 1:i + 1]
        if not np.any(np.isnan(window)):
            current = bb_width[i]
            percentile[i] = np.sum(window <= current) / len(window) * 100.0
    
    return percentile

def calculate_atr_ratio(atr, short_period=7, long_period=30):
    """ATR ratio for vol spike detection - high ratio = vol expansion"""
    n = len(atr)
    ratio = np.full(n, np.nan)
    
    atr_short = pd.Series(atr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(atr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    mask = atr_long > 1e-10
    ratio[mask] = atr_short[mask] / atr_long[mask]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs for regime filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=60)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(atr, short_period=7, long_period=30)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
            continue
        if np.isnan(bb_width[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO REGIME FILTERS ===
        # 1w HMA = ultra-long bias
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # 1d HMA = medium-term regime (asymmetric logic)
        regime_bull = close[i] > hma_1d_aligned[i]
        regime_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME ===
        # BB Width percentile < 20 = squeeze (vol compression)
        # BB Width percentile > 80 = expansion (vol high)
        bb_squeeze = bb_width_pct[i] < 25.0
        bb_expansion = bb_width_pct[i] > 75.0
        
        # ATR ratio > 1.5 = vol spike (expansion phase)
        # ATR ratio < 0.8 = vol crush (reversion opportunity)
        vol_spike = atr_ratio[i] > 1.5
        vol_crush = atr_ratio[i] < 0.85
        
        # === RSI MOMENTUM (Asymmetric thresholds) ===
        # In bull regime: enter long on RSI pullback (40-50)
        # In bear regime: enter short on RSI bounce (50-60)
        rsi_long_pullback = 35.0 < rsi[i] < 55.0
        rsi_short_bounce = 45.0 < rsi[i] < 65.0
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === DESIRED SIGNAL - ASYMMETRIC REGIME LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES (only in bull regime or macro bull)
        if regime_bull or macro_bull:
            # Path 1: BB squeeze + vol spike + RSI pullback (breakout from compression)
            if bb_squeeze and vol_spike and rsi_long_pullback:
                desired_signal = BASE_SIZE
            # Path 2: Vol crush + RSI oversold (mean reversion in bull)
            elif vol_crush and rsi_oversold:
                desired_signal = BASE_SIZE * 0.5
            # Path 3: Price > both HMAs + RSI momentum (trend continuation)
            elif close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i] and rsi[i] > 50.0:
                desired_signal = BASE_SIZE * 0.5
            # Path 4: BB expansion + RSI strong (momentum breakout)
            elif bb_expansion and rsi[i] > 55.0:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRIES (only in bear regime or macro bear)
        elif regime_bear or macro_bear:
            # Path 1: BB squeeze + vol spike + RSI bounce (breakdown from compression)
            if bb_squeeze and vol_spike and rsi_short_bounce:
                desired_signal = -BASE_SIZE
            # Path 2: Vol crush + RSI overbought (mean reversion in bear)
            elif vol_crush and rsi_overbought:
                desired_signal = -BASE_SIZE * 0.5
            # Path 3: Price < both HMAs + RSI weak (trend continuation)
            elif close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i] and rsi[i] < 50.0:
                desired_signal = -BASE_SIZE * 0.5
            # Path 4: BB expansion + RSI weak (momentum breakdown)
            elif bb_expansion and rsi[i] < 45.0:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if abs(desired_signal) >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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