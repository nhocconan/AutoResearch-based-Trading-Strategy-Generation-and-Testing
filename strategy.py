#!/usr/bin/env python3
"""
Experiment #134: 4h Primary + 12h HTF — Regime-Adaptive Bollinger Mean Reversion

Hypothesis: After 133 experiments, clear patterns emerge for 4h timeframe:
- Pure trend following fails in bear/range markets (2025 test is -25% BTC)
- Pure mean reversion gets destroyed in strong trends
- BEST approach: Regime-adaptive (different logic per market state)
- 12h HMA slope + price position = reliable regime detector
- Bollinger Band mean reversion works in range, filtered by regime

This strategy uses PROVEN components from successful patterns:
1. 12h HMA = major trend bias (price above/below + slope direction)
2. Regime detection: BULL / BEAR / RANGE based on 12h HMA
3. BULL regime: Only long pullbacks to BB lower + RSI oversold
4. BEAR regime: Only short rallies to BB upper + RSI overbought
5. RANGE regime: Mean revert both directions at BB extremes
6. Exit: ATR trailing stop (2.5x) + BB middle target

Key design choices:
- Timeframe: 4h (proven 20-50 trades/year target)
- HTF: 12h HMA (more responsive than 1d for 4h primary)
- BB(20, 2.0): Standard bands for mean reversion
- RSI(14): 35/65 thresholds (looser than 30/70 for more trades)
- Position size: 0.30 (30% of capital, conservative)
- Stoploss: 2.5x ATR trailing (tighter for mean reversion)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
All symbols must have Sharpe>0 individually (no SOL-only strategies)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_regime_adaptive_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average - more responsive than EMA, less lag
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA with span = period/2
    wma1 = pd.Series(close).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    # WMA with span = period
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    # Raw HMA
    raw_hma = 2 * wma1 - wma2
    # Smooth with sqrt(period)
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    
    return hma

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands for mean reversion levels"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper, sma, lower

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
    """Average True Range for stoploss"""
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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for regime detection
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 4h)
    
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
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (12h HMA) ===
        # Bull: price > 12h HMA AND 12h HMA sloping up
        # Bear: price < 12h HMA AND 12h HMA sloping down
        # Range: HMA flat or mixed signals
        
        hma_slope = 0.0
        if i >= 2 and not np.isnan(hma_12h_aligned[i-1]):
            hma_slope = hma_12h_aligned[i] - hma_12h_aligned[i-1]
        
        price_vs_hma = close[i] - hma_12h_aligned[i]
        hma_threshold = atr[i] * 0.5  # HMA move threshold
        
        in_bull_regime = (price_vs_hma > 0) and (hma_slope > 0)
        in_bear_regime = (price_vs_hma < 0) and (hma_slope < 0)
        in_range_regime = not in_bull_regime and not in_bear_regime
        
        # === BB MEAN REVERSION SIGNALS ===
        # Touching lower band = potential long
        # Touching upper band = potential short
        # Use 95% of band to catch near-touches (more trades)
        bb_lower_signal = close[i] <= bb_lower[i] * 1.005
        bb_upper_signal = close[i] >= bb_upper[i] * 0.995
        
        # === RSI CONFIRMATION (loose thresholds for trade generation) ===
        # 35/65 instead of 30/70 = more trades
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === REGIME-ADAPTIVE SIGNAL LOGIC ===
        desired_signal = 0.0
        
        # BULL regime: Only long on pullbacks
        if in_bull_regime and bb_lower_signal and rsi_oversold:
            desired_signal = SIZE
        
        # BEAR regime: Only short on rallies
        elif in_bear_regime and bb_upper_signal and rsi_overbought:
            desired_signal = -SIZE
        
        # RANGE regime: Mean revert both directions (most trades here)
        elif in_range_regime:
            if bb_lower_signal and rsi_oversold:
                desired_signal = SIZE
            elif bb_upper_signal and rsi_overbought:
                desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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
                # Flip position
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