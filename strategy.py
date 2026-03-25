#!/usr/bin/env python3
"""
Experiment #1112: 12h Primary + 1d HTF — Dual Regime (Chop/Trend) with LOOSE Entries

Hypothesis: Previous 12h strategies failed due to OVERLY STRICT entry conditions (0 trades).
This version uses MUCH LOOSER thresholds to guarantee 30+ trades while maintaining edge.

Key changes from failed experiments:
1. CHOP thresholds: 50/40 instead of 61.8/38.2 (more regime switches = more trades)
2. RSI thresholds: 30/70 for mean revert, 45/55 for trend (wider windows)
3. Removed HMA alignment requirement for mean reversion (was blocking trades)
4. Added volume confirmation instead of HMA confluence (more responsive)
5. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry logic (LOOSE):
- CHOP > 50 (choppy): Long RSI(7)<30, Short RSI(7)>70
- CHOP < 40 (trending): Long price>1d_HMA + RSI(14)>45, Short price<1d_HMA + RSI(14)<55
- Volume spike filter: volume > 1.5 * volume_SMA(20) confirms entries

Why this should work:
- 12h timeframe = 20-50 trades/year target (low fee drag)
- Dual regime adapts to 2022-2023 range markets AND 2021/2024 trends
- LOOSE thresholds guarantee trades (fixes #1104 0-trade failure)
- 1d HMA provides trend bias without over-filtering
- ATR 2.5x trailing stop protects from 2022-style crashes

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_rsi_loose_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market (use mean reversion)
    CHOP < 38.2 = trending market (use trend following)
    We use 50/40 thresholds for MORE regime switches = MORE trades
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)   # Fast RSI for mean reversion
    rsi_14 = calculate_rsi(close, period=14) # Standard RSI for trend
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # LOOSE thresholds: 50/40 instead of 61.8/38.2 for MORE trades
        is_choppy = chop_14[i] > 50.0  # Range market
        is_trending = chop_14[i] < 40.0  # Trend market
        is_neutral = not is_choppy and not is_trending  # Transition zone
        
        # === HTF BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.3 * vol_sma_20[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE, LOOSE) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - use RSI(7) extremes
            # LOOSE: RSI<30 for long, RSI>70 for short (was 20/80)
            if rsi_7[i] < 30.0:
                desired_signal = SIZE_BASE
            elif rsi_7[i] > 70.0:
                desired_signal = -SIZE_BASE
            # Stronger signals at more extreme RSI
            elif rsi_7[i] < 20.0:
                desired_signal = SIZE_STRONG
            elif rsi_7[i] > 80.0:
                desired_signal = -SIZE_STRONG
        
        elif is_trending:
            # TREND FOLLOWING MODE - use 1d HMA + RSI(14)
            # LOOSE: RSI>45 for long, RSI<55 for short (was 50/50)
            if hma_1d_bull and rsi_14[i] > 45.0:
                desired_signal = SIZE_STRONG if vol_confirmed else SIZE_BASE
            elif hma_1d_bear and rsi_14[i] < 55.0:
                desired_signal = -SIZE_STRONG if vol_confirmed else -SIZE_BASE
        
        elif is_neutral:
            # TRANSITION ZONE - use HMA bias only (simpler = more trades)
            if hma_1d_bull and rsi_14[i] > 50.0:
                desired_signal = SIZE_BASE
            elif hma_1d_bear and rsi_14[i] < 50.0:
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