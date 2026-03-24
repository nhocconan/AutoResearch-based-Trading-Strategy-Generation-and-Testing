#!/usr/bin/env python3
"""
Experiment #608: 4h Primary + 12h/1d HTF — Simplified HMA Trend + RSI Pullback + Choppiness Filter

Hypothesis: After 517 failed experiments, the key insight is SIMPLICITY + TRADE FREQUENCY.
Most recent failures (#596-#607) got 0 trades due to over-filtering. This strategy:
1. Uses proven HMA trend following (worked in best strategies)
2. RSI pullback entries (not extremes - ensures we get trades)
3. Choppiness as soft filter (not hard block)
4. 12h HMA for trend bias + 1d HMA for macro confirmation
5. Relaxed entry conditions to guarantee 20-50 trades/year

Key differences from failed #602 (mtf_4h_crsi_chop_hma_kama_1d_v2):
1. SIMPLER logic - HMA crossover instead of complex KAMA+CRSI
2. RSI 40-60 pullback zone (not extremes) = more entries
3. Choppiness reduces size but doesn't block entries
4. No ADX filter (was blocking too many trades)
5. Clearer regime: just HTF alignment + HMA direction

Strategy logic:
1. 1d HMA(21) = macro trend (price above = bull bias)
2. 12h HMA(21) = medium trend (price above = confirm bias)
3. 4h HMA(16/48) crossover = entry trigger
4. 4h RSI(14) 40-60 = pullback entry zone (not extremes)
5. 4h Choppiness(14) = size modifier (chop = reduce size 20%)
6. ATR(14)*2.5 stoploss on all positions

Entry conditions (OR logic for more trades):
- LONG: 1d HMA bull + 12h HMA bull + 4h HMA16>48 + RSI 40-60
- SHORT: 1d HMA bear + 12h HMA bear + 4h HMA16<48 + RSI 40-60

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h1d_simple_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for medium trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_REDUCED = 0.20
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d macro + 12h medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_12h_aligned[i]
        
        # === HMA TREND (4h) ===
        hma_bull = hma_16[i] > hma_48[i]
        hma_bear = hma_16[i] < hma_48[i]
        
        # HMA slope confirmation (look back 5 bars)
        hma_slope_bull = hma_16[i] > hma_16[i-5] if i >= 5 and not np.isnan(hma_16[i-5]) else True
        hma_slope_bear = hma_16[i] < hma_16[i-5] if i >= 5 and not np.isnan(hma_16[i-5]) else True
        
        # === RSI PULLBACK ZONE (40-60, not extremes) ===
        rsi_pullback_long = 40.0 <= rsi[i] <= 60.0
        rsi_pullback_short = 40.0 <= rsi[i] <= 60.0
        
        # RSI momentum (rising for long, falling for short)
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # === CHOPPINESS REGIME (soft filter) ===
        chop_trending = chop[i] < 50.0  # More trending
        chop_chopping = chop[i] >= 50.0  # More choppy
        
        # === ENTRY LOGIC (OR conditions for more trades) ===
        desired_signal = 0.0
        signal_strength = 1.0
        
        # LONG entries
        if htf_bull and hma_bull and hma_slope_bull:
            if rsi_pullback_long:
                desired_signal = SIZE_BASE
                if rsi_rising:
                    desired_signal = SIZE_STRONG
            elif rsi[i] < 40.0 and rsi_rising:
                # Oversold recovery
                desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_bear and hma_bear and hma_slope_bear:
            if rsi_pullback_short:
                desired_signal = -SIZE_BASE
                if rsi_falling:
                    desired_signal = -SIZE_STRONG
            elif rsi[i] > 60.0 and rsi_falling:
                # Overbought rejection
                desired_signal = -SIZE_BASE
        
        # Apply choppiness modifier (reduce size in chop, don't block)
        if chop_chopping and desired_signal != 0:
            signal_strength = 0.8
        
        desired_signal = desired_signal * signal_strength
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_REDUCED * 0.9:
            final_signal = np.sign(desired_signal) * SIZE_REDUCED
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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