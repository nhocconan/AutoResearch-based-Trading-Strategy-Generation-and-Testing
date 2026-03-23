#!/usr/bin/env python3
"""
Experiment #334: 4h Primary + 12h/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous failures came from over-filtering (0 trades) or complex regimes.
This strategy uses PROVEN patterns with SIMPLER logic:
1. 4h HMA(21) for primary trend direction
2. 12h HMA(21) for intermediate bias (not hard filter)
3. 1d HMA(21) for macro regime (bull/bear classification)
4. RSI(14) pullback entries in trend direction (not extremes)
5. ATR(14) trailing stoploss only (no complex take-profit)

KEY INSIGHT: Simpler = more trades = better statistics. Previous complex regimes
got 0 trades. This uses continuous signal generation with clear entry/exit.

TARGET: 30-50 trades/year on 4h, Sharpe > 0.7 on ALL symbols, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h1d_bias_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA) - faster response than EMA."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index for regime detection."""
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_21_4h = calculate_hma(close, 21)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align HTF HMAs
    hma_21_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h_raw)
    
    hma_21_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d_raw)
    
    signals = np.zeros(n)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Position sizing levels
    SIZE_STRONG = 0.30  # High confidence
    SIZE_WEAK = 0.15    # Lower confidence
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_21_4h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_21_12h_aligned[i]) or np.isnan(hma_21_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO REGIME (1d HMA) ===
        bull_macro = close[i] > hma_21_1d_aligned[i]
        bear_macro = close[i] < hma_21_1d_aligned[i]
        
        # === INTERMEDIATE BIAS (12h HMA) ===
        bull_12h = close[i] > hma_21_12h_aligned[i]
        bear_12h = close[i] < hma_21_12h_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        bull_4h = close[i] > hma_21_4h[i]
        bear_4h = close[i] < hma_21_4h[i]
        
        # === REGIME (Choppiness) ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG entries: RSI pullback in uptrend
        if bull_4h and rsi_14[i] >= 40.0 and rsi_14[i] <= 60.0:
            if bull_macro and bull_12h:
                desired_signal = SIZE_STRONG  # All aligned bull
            elif bull_macro or bull_12h:
                desired_signal = SIZE_WEAK  # Partial alignment
            elif is_choppy and rsi_14[i] <= 45.0:
                desired_signal = SIZE_WEAK  # Mean reversion long in chop
        
        # SHORT entries: RSI pullback in downtrend
        elif bear_4h and rsi_14[i] >= 40.0 and rsi_14[i] <= 60.0:
            if bear_macro and bear_12h:
                desired_signal = -SIZE_STRONG  # All aligned bear
            elif bear_macro or bear_12h:
                desired_signal = -SIZE_WEAK  # Partial alignment
            elif is_choppy and rsi_14[i] >= 55.0:
                desired_signal = -SIZE_WEAK  # Mean reversion short in chop
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend flips bearish
        if in_position and position_side > 0 and bear_4h and rsi_14[i] < 45.0:
            desired_signal = 0.0
        
        # Exit short if 4h trend flips bullish
        if in_position and position_side < 0 and bull_4h and rsi_14[i] > 55.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still intact
                if bull_4h or (is_choppy and rsi_14[i] >= 40.0):
                    desired_signal = SIZE_STRONG if bull_macro else SIZE_WEAK
            elif position_side < 0:
                # Hold short if trend still intact
                if bear_4h or (is_choppy and rsi_14[i] <= 60.0):
                    desired_signal = -SIZE_STRONG if bear_macro else -SIZE_WEAK
        
        # === UPDATE POSITION TRACKING ===
        prev_side = position_side
        
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals