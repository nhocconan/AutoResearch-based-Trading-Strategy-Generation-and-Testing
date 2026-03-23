#!/usr/bin/env python3
"""
Experiment #1144: 4h Primary + 1d/1w HTF — Choppiness Regime + RSI + HMA Trend

Hypothesis: After 833+ failed experiments, the key insight is REGIME ADAPTATION.
Simple trend strategies fail in bear/range markets (2022 crash, 2025 bear).
This strategy uses Choppiness Index to switch between:
1. CHOP > 61.8 (range): Mean reversion at RSI extremes with tight stops
2. CHOP < 38.2 (trend): Trend following with HMA direction + RSI pullback
3. CHOP 38-62 (transition): Stay flat or reduce size

Key improvements over #1129:
- Choppiness regime filter (proven 0.75+ Sharpe in research)
- LOOSER RSI thresholds (40/60 not 45/55) to ensure trade frequency
- 1w HMA as additional macro filter (works across all symbols)
- Asymmetric sizing: 0.30 for trend, 0.20 for mean-revert (less risk in chop)
- Hold logic maintains through regime continuity (not just RSI)

Timeframe: 4h (primary)
HTF: 1d + 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.20-0.30 discrete (minimize fee churn)
Stoploss: 2.5x ATR trailing
Target: 25-50 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_rsi_hma_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppiness vs trending.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = Range/Chop (mean reversion works)
    CHOP < 38.2 = Trend (trend following works)
    38.2 < CHOP < 61.8 = Transition (reduce size or stay flat)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    TREND_SIZE = 0.30  # Size for trending regime
    CHOP_SIZE = 0.20   # Reduced size for choppy regime (less risk)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_4h[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_high = chop[i] > 61.8  # Range/chop regime
        chop_low = chop[i] < 38.2   # Trend regime
        chop_mid = not chop_high and not chop_low  # Transition
        
        # === MACRO TREND (1d + 1w HMA) ===
        # Both must agree for strong signal
        macro_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        
        # === 4h TREND DIRECTION ===
        trend_bull = close[i] > hma_4h[i] and hma_4h[i] > hma_4h[i-1] if not np.isnan(hma_4h[i-1]) else close[i] > hma_4h[i]
        trend_bear = close[i] < hma_4h[i] and hma_4h[i] < hma_4h[i-1] if not np.isnan(hma_4h[i-1]) else close[i] < hma_4h[i]
        
        # === RSI SIGNALS (LOOSE thresholds for trade frequency) ===
        rsi_oversold = rsi_4h[i] < 40.0  # Mean reversion long
        rsi_overbought = rsi_4h[i] > 60.0  # Mean reversion short
        rsi_pullback_long = rsi_4h[i] < 50.0  # Trend pullback long
        rsi_pullback_short = rsi_4h[i] > 50.0  # Trend pullback short
        
        desired_signal = 0.0
        current_size = TREND_SIZE if chop_low else CHOP_SIZE if chop_high else 0.0
        
        # === TREND REGIME (CHOP < 38.2) ===
        if chop_low:
            # Long: Macro bull + 4h trend bull + RSI pullback
            if macro_bull and trend_bull and rsi_pullback_long:
                desired_signal = TREND_SIZE
            # Short: Macro bear + 4h trend bear + RSI pullback
            elif macro_bear and trend_bear and rsi_pullback_short:
                desired_signal = -TREND_SIZE
        
        # === CHOP REGIME (CHOP > 61.8) ===
        elif chop_high:
            # Mean reversion long: RSI oversold + price > 1d HMA (not in crash)
            if rsi_oversold and close[i] > hma_1d_aligned[i]:
                desired_signal = CHOP_SIZE
            # Mean reversion short: RSI overbought + price < 1d HMA
            elif rsi_overbought and close[i] < hma_1d_aligned[i]:
                desired_signal = -CHOP_SIZE
        
        # === TRANSITION REGIME (38.2 < CHOP < 61.8) ===
        # Stay flat or hold existing position with reduced conviction
        elif chop_mid:
            if in_position:
                # Hold existing but don't add
                desired_signal = position_side * CHOP_SIZE
            else:
                desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull OR in chop regime with RSI not extreme
                if (macro_bull and chop_low) or (chop_high and rsi_4h[i] < 70):
                    desired_signal = position_side * current_size
            elif position_side < 0:
                # Hold short if macro still bear OR in chop regime with RSI not extreme
                if (macro_bear and chop_low) or (chop_high and rsi_4h[i] > 30):
                    desired_signal = position_side * current_size
        
        # === EXIT CONDITIONS ===
        # Exit when macro trend reverses strongly
        if in_position and position_side > 0:
            # Exit long if macro reverses to bear AND chop is low (trending down)
            if macro_bear and chop_low:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses to bull AND chop is low (trending up)
            if macro_bull and chop_low:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.15:
            desired_signal = current_size
        elif desired_signal < -0.15:
            desired_signal = -current_size
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals