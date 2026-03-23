#!/usr/bin/env python3
"""
Experiment #386: 12h Primary + 1d HTF — Dual Regime HMA + Choppiness + RSI

Hypothesis: Previous 12h strategies over-filtered with too many confluence conditions.
This strategy uses SIMPLIFIED dual-regime logic with relaxed thresholds to ensure
trade generation while maintaining quality. Key innovations:

1. HMA(21/63) for primary trend - faster response than KAMA, proven on 12h
2. Choppiness Index regime: CHOP>61.8=range(mean-revert), CHOP<38.2=trend(breakout)
3. Relaxed RSI thresholds: <35 for long, >65 for short (not 30/70 which filter too much)
4. Single HTF (1d HMA) for bias - soft filter, not hard requirement
5. Donchian(20) breakout for trend continuation entries
6. Position size 0.28 discrete - balances return vs drawdown on 12h

Target: 25-45 trades/year on 12h, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL individually).
Must beat current best: mtf_4h_triple_regime_crsi_donchian_1d1w_v1 (Sharpe=0.612)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_chop_rsi_donchian_1d_dual_regime_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
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
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period=period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(n)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_21 = calculate_hma(close, period=21)
    hma_63 = calculate_hma(close, period=63)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 12h (target 25-45 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_63[i]):
            continue
        
        # === HTF BIAS (1d HMA) - SOFT FILTER ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA) ===
        hma_bullish = hma_21[i] > hma_63[i]
        hma_bearish = hma_21[i] < hma_63[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_ranging = chop_14[i] > 55.0  # Relaxed from 61.8
        is_trending = chop_14[i] < 45.0  # Relaxed from 38.2
        
        # === RSI EXTREMES (Relaxed thresholds) ===
        rsi_oversold = rsi_14[i] < 40.0  # Relaxed from 35
        rsi_overbought = rsi_14[i] > 60.0  # Relaxed from 65
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - Multiple entry paths for trade generation
        if hma_bullish or price_above_hma_1d:
            # Path 1: Trend breakout (trending regime)
            if is_trending and breakout_long:
                desired_signal = BASE_SIZE
            # Path 2: Mean reversion pullback (ranging regime)
            elif is_ranging and rsi_oversold:
                desired_signal = BASE_SIZE
            # Path 3: HMA crossover confirmation
            elif hma_bullish and rsi_oversold:
                desired_signal = BASE_SIZE
            # Path 4: Simple RSI oversold with HTF support
            elif rsi_oversold and price_above_hma_1d:
                desired_signal = BASE_SIZE
        
        # SHORT SETUP - Multiple entry paths for trade generation
        if hma_bearish or price_below_hma_1d:
            # Path 1: Trend breakdown (trending regime)
            if is_trending and breakout_short:
                desired_signal = -BASE_SIZE
            # Path 2: Mean reversion rally (ranging regime)
            elif is_ranging and rsi_overbought:
                desired_signal = -BASE_SIZE
            # Path 3: HMA crossover confirmation
            elif hma_bearish and rsi_overbought:
                desired_signal = -BASE_SIZE
            # Path 4: Simple RSI overbought with HTF resistance
            elif rsi_overbought and price_below_hma_1d:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === RSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and rsi_14[i] > 65:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 35:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_1d and hma_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and hma_bullish:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or price_above_hma_1d):
                desired_signal = BASE_SIZE
            elif position_side < 0 and (hma_bearish or price_below_hma_1d):
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals