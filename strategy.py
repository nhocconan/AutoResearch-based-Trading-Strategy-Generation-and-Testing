#!/usr/bin/env python3
"""
Experiment #959: 1h Primary + 4h/12h HTF — Simplified Trend + Mean Reversion

Hypothesis: Previous strategies failed due to TOO STRICT entry conditions (0 trades).
This strategy uses SIMPLER logic with LOOSER thresholds to guarantee trade frequency.

Key changes from failures:
1. SINGLE HTF bias (4h HMA only, not 1d+1w which is too restrictive)
2. SIMPLE RSI(14) instead of complex CRSI (RSI<35/>65 vs CRSI<10/>90)
3. CHOP filter as bonus not requirement (don't block trades if CHOP neutral)
4. Volume filter: simple 20-bar SMA, not complex ratios
5. Session filter: 06-22 UTC (wider than 08-20 to catch more opportunities)

Entry logic (LOOSE to guarantee 40-80 trades/year):
- LONG: 4h HMA bull + RSI(14)<40 OR RSI(14)<50 + price pullback to EMA21
- SHORT: 4h HMA bear + RSI(14)>60 OR RSI(14)>50 + price rally to EMA21
- CHOP>60 = increase size slightly (range = better mean reversion)

Why this should work:
- 4h HMA gives trend bias without over-filtering
- RSI(14) extremes happen frequently enough for trade frequency
- 1h TF captures intraday swings without 15m noise
- ATR stoploss protects from 2022-style crashes

Target: Sharpe>0.45, trades>=40/year, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_simple_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - fast responsive trend"""
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
    
    return wma(diff, sqrt_n)

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stops"""
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
    """Relative Strength Index - simple and reliable"""
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
    """Choppiness Index - regime detection"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        
        if highest_high > lowest_low and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / tr_sum) / np.log10(period)
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple volume SMA for volume confirmation"""
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
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    hma_1h_21 = calculate_hma(close, period=21)
    ema_1h_21 = calculate_ema(close, period=21)
    ema_1h_50 = calculate_ema(close, period=50)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_1h_21[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA - primary, 12h HMA - confirmation) ===
        # Use 4h as primary trend, 12h as secondary confirmation (not requirement)
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both agree, weak when only 4h agrees
        htf_strong_bull = htf_4h_bull and htf_12h_bull
        htf_strong_bear = htf_4h_bear and htf_12h_bear
        htf_weak_bull = htf_4h_bull and not htf_12h_bull
        htf_weak_bear = htf_4h_bear and not htf_12h_bull
        
        # === REGIME DETECTION (CHOP) ===
        is_ranging = chop_14[i] > 55.0  # Lower threshold for more range detection
        is_trending = chop_14[i] < 45.0  # Lower threshold for more trend detection
        
        # === RSI CONDITIONS (LOOSE THRESHOLDS FOR TRADES) ===
        rsi_oversold = rsi_14[i] < 40.0  # Much looser than CRSI<10
        rsi_overbought = rsi_14[i] > 60.0  # Much looser than CRSI>90
        rsi_neutral_low = rsi_14[i] < 50.0
        rsi_neutral_high = rsi_14[i] > 50.0
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > vol_sma_20[i] * 0.8  # At least 80% of avg volume
        
        # === PRICE POSITION vs EMA21 ===
        price_below_ema = close[i] < ema_1h_21[i]
        price_above_ema = close[i] > ema_1h_21[i]
        
        # === HOUR SESSION FILTER (06-22 UTC for liquidity) ===
        # Extract hour from open_time (assuming milliseconds timestamp)
        hour = (prices['open_time'].iloc[i] // 3600000) % 24
        session_ok = 6 <= hour <= 22
        
        # === ENTRY LOGIC (SIMPLIFIED - LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        # LONG entries - multiple paths to entry
        if htf_4h_bull:  # Only require 4h bull, not both
            # Path 1: Strong trend + RSI pullback
            if htf_strong_bull and rsi_oversold and volume_ok:
                desired_signal = SIZE_STRONG
            # Path 2: Weak trend + deep RSI + price below EMA (pullback)
            elif htf_weak_bull and rsi_14[i] < 35.0 and price_below_ema:
                desired_signal = SIZE_BASE
            # Path 3: Range regime + RSI oversold (mean reversion)
            elif is_ranging and rsi_oversold:
                desired_signal = SIZE_BASE
            # Path 4: Any bull + very oversold RSI
            elif rsi_14[i] < 30.0:
                desired_signal = SIZE_BASE
        
        # SHORT entries - multiple paths to entry
        elif htf_4h_bear:  # Only require 4h bear, not both
            # Path 1: Strong trend + RSI rally
            if htf_strong_bear and rsi_overbought and volume_ok:
                desired_signal = -SIZE_STRONG
            # Path 2: Weak trend + high RSI + price above EMA (rally)
            elif htf_weak_bear and rsi_14[i] > 65.0 and price_above_ema:
                desired_signal = -SIZE_BASE
            # Path 3: Range regime + RSI overbought (mean reversion)
            elif is_ranging and rsi_overbought:
                desired_signal = -SIZE_BASE
            # Path 4: Any bear + very overbought RSI
            elif rsi_14[i] > 70.0:
                desired_signal = -SIZE_BASE
        
        # Apply session filter (reduce size outside session, don't block)
        if not session_ok and desired_signal != 0.0:
            desired_signal = desired_signal * 0.5  # Half size outside session
        
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