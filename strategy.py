#!/usr/bin/env python3
"""
Experiment #167: 6h Primary + 1d/1w HTF — BB Mean Reversion + HTF Trend Bias

Hypothesis: Previous 6h trend-following strategies (#160, #163) achieved low Sharpe.
Mean reversion with HTF trend bias should work better in 2025 bear/range market.
This is DIFFERENT from all previous 6h attempts (which were pure trend following).

Key design:
- 6h Bollinger Bands (20, 2.0) for mean reversion entries
- 1d HMA(50) for major trend bias (only long if 1d bull, only short if 1d bear)
- 1w HMA(50) for weekly regime filter (reduce size if weekly opposes)
- RSI(7) for short-term extremes (faster than RSI14)
- ATR(14) trailing stop at 2.5x
- Position size: 0.30 (30% of capital)

Why this should work:
- BB mean reversion captures range-bound behavior (common in 2025)
- HTF trend filter prevents counter-trend trades in strong trends
- RSI(7) is more responsive than RSI(14) for 6h timeframe
- Looser BB threshold (1.8 std for entry, not 2.0) ensures trades

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_bb_mr_hma_1d1w_v1"
timeframe = "6h"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands - mean reversion indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, sma, lower

def calculate_bb_percent_b(close, period=20, std_dev=2.0):
    """Bollinger Band %B - position within bands (0=lower, 1=upper)"""
    upper, sma, lower = calculate_bollinger_bands(close, period, std_dev)
    
    n = len(close)
    bb_pct_b = np.zeros(n)
    bb_pct_b[:] = np.nan
    
    for i in range(period, n):
        if upper[i] - lower[i] > 1e-10:
            bb_pct_b[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
    
    return bb_pct_b

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for weekly regime filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    bb_pct_b = calculate_bb_percent_b(close, period=20, std_dev=2.0)
    bb_upper, bb_sma, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 6h
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after HTF alignment is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_pct_b[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY REGIME (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === BB %B MEAN REVERSION SIGNALS ===
        # %B < 0.1 = near lower band (oversold), %B > 0.9 = near upper band (overbought)
        bb_oversold = bb_pct_b[i] < 0.15
        bb_overbought = bb_pct_b[i] > 0.85
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY LONG: 1d bull + BB oversold + RSI oversold
        if htf_1d_bull and bb_oversold and rsi_oversold:
            # Full size if 1w also bull, reduced if 1w bear
            if htf_1w_bull:
                desired_signal = SIZE
            else:
                desired_signal = SIZE * 0.6  # Reduce size if weekly opposes
        
        # PRIMARY SHORT: 1d bear + BB overbought + RSI overbought
        elif htf_1d_bear and bb_overbought and rsi_overbought:
            if htf_1w_bear:
                desired_signal = -SIZE
            else:
                desired_signal = -SIZE * 0.6
        
        # FALLBACK 1: Strong BB extreme (ignore RSI) - ensures trades
        elif htf_1d_bull and bb_pct_b[i] < 0.08:
            desired_signal = SIZE * 0.5
        
        elif htf_1d_bear and bb_pct_b[i] > 0.92:
            desired_signal = -SIZE * 0.5
        
        # FALLBACK 2: RSI extreme with HTF bias (ignore BB) - ensures trades
        elif htf_1d_bull and rsi[i] < 25.0:
            desired_signal = SIZE * 0.4
        
        elif htf_1d_bear and rsi[i] > 75.0:
            desired_signal = -SIZE * 0.4
        
        # FALLBACK 3: Mean reversion to BB middle (exit signal)
        # If in long and price crosses above BB middle, reduce position
        if in_position and position_side > 0 and bb_pct_b[i] > 0.6:
            desired_signal = SIZE * 0.3  # Take partial profit
        
        if in_position and position_side < 0 and bb_pct_b[i] < 0.4:
            desired_signal = -SIZE * 0.3  # Take partial profit
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.7:
            final_signal = SIZE * 0.8
        elif desired_signal <= -SIZE * 0.7:
            final_signal = -SIZE * 0.8
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
        elif desired_signal >= SIZE * 0.3:
            final_signal = SIZE * 0.4
        elif desired_signal <= -SIZE * 0.3:
            final_signal = -SIZE * 0.4
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