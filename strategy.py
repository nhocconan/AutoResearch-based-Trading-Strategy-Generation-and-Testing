#!/usr/bin/env python3
"""
Experiment #1658: 30m Primary + 4h/1d HTF — Regime-Adaptive with Loose Entries

Hypothesis: Previous 30m failures (#1648, #1655) got Sharpe=0.000 due to OVER-FILTERING.
Too many confluence conditions = 0 trades. This strategy uses LOOSE thresholds while
keeping the proven MTF structure: 4h for TREND DIRECTION, 30m for ENTRY TIMING.

Key changes from failures:
1. LOOSER RSI thresholds: 35/65 not 30/70 or 20/80 (more triggers)
2. LOOSER CHOP regime: >50 choppy, <50 trending (simpler binary)
3. 4h HMA for direction ONLY (not entry trigger)
4. Volume filter: >0.7x avg (not 1.0x - too strict)
5. Session filter REMOVED (was killing trades in #1648)
6. Smaller position size: 0.25 base (30m needs smaller than 1d)

Why 30m with 4h HTF:
- 4h trend = signal direction (trade WITH 4h trend only)
- 30m RSI/CHOP = entry timing within 4h trend
- Target: 40-80 trades/year (0.8-1.5/week)

Entry Logic:
- 4h HMA(21) > price = bearish bias (only short or flat)
- 4h HMA(21) < price = bullish bias (only long or flat)
- 30m CHOP > 50 = mean revert (RSI extremes trigger)
- 30m CHOP < 50 = trend follow (price vs 30m HMA)
- Volume > 0.7x 20-bar avg (confirmation)

Risk: 2.5x ATR trailing stop, discrete signal levels (0.0, ±0.20, ±0.25)

Target: Sharpe > 0.3, trades > 40/symbol train, > 5/symbol test, DD > -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_rsi_hma_4h_loose_entry_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if loss_smooth[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i-1] / loss_smooth[i-1]))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 50 = choppy/range (mean revert)
    CHOP < 50 = trending (trend follow)
    Simplified binary regime for more triggers
    """
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
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for stronger trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    hma_30m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_30m[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h + 1d confluence) ===
        # Only trade WITH the HTF trend direction
        bullish_4h = close[i] > hma_4h_aligned[i]
        bearish_4h = close[i] < hma_4h_aligned[i]
        
        bullish_1d = close[i] > hma_1d_aligned[i]
        bearish_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias: both 4h and 1d agree
        strong_bull = bullish_4h and bullish_1d
        strong_bear = bearish_4h and bearish_1d
        
        # Weak bias: only 4h agrees (1d neutral or opposite)
        weak_bull = bullish_4h and not bearish_1d
        weak_bear = bearish_4h and not bullish_1d
        
        # === REGIME DETECTION (Simplified Choppiness) ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] <= 50.0
        
        # === VOLUME FILTER (LOOSE: >0.7x avg) ===
        vol_confirmed = volume[i] > 0.7 * vol_avg[i] if not np.isnan(vol_avg[i]) else True
        
        # === DESIRED SIGNAL BASED ON REGIME + HTF BIAS ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - use RSI extremes (LOOSE thresholds)
            # Long: RSI < 40 (oversold) + bullish HTF bias
            if rsi[i] < 40.0:
                if strong_bull:
                    signal_strength = BASE_SIZE
                elif weak_bull:
                    signal_strength = REDUCED_SIZE
                if signal_strength > 0:
                    desired_signal = signal_strength if vol_confirmed else 0.0
            
            # Short: RSI > 60 (overbought) + bearish HTF bias
            elif rsi[i] > 60.0:
                if strong_bear:
                    signal_strength = BASE_SIZE
                elif weak_bear:
                    signal_strength = REDUCED_SIZE
                if signal_strength > 0:
                    desired_signal = -signal_strength if vol_confirmed else 0.0
        
        elif is_trending:
            # TREND REGIME - use HMA position + HTF bias
            # Long: Price > 30m HMA + bullish HTF
            if close[i] > hma_30m[i]:
                if strong_bull:
                    signal_strength = BASE_SIZE
                elif weak_bull:
                    signal_strength = REDUCED_SIZE
                if signal_strength > 0:
                    desired_signal = signal_strength if vol_confirmed else 0.0
            
            # Short: Price < 30m HMA + bearish HTF
            elif close[i] < hma_30m[i]:
                if strong_bear:
                    signal_strength = BASE_SIZE
                elif weak_bear:
                    signal_strength = REDUCED_SIZE
                if signal_strength > 0:
                    desired_signal = -signal_strength if vol_confirmed else 0.0
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
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