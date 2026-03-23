#!/usr/bin/env python3
"""
Experiment #407: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 341 failed strategies, complexity is the enemy. This uses
proven components with MINIMAL filters to ensure trades actually execute.

Key learnings from failures:
- #395, #398, #401, #405: Sharpe=0.000 (ZERO TRADES) - too many filters
- #397, #403: 1d strategies got Sharpe 0.296-0.313 with simpler logic
- Current best #394: Sharpe=0.612 with triple regime (complex)

New approach for 1d:
1. HMA(21/63) crossover - proven in mtf_hma_rsi_zscore_v1 (Sharpe=5.4)
2. RSI(14) with MODERATE thresholds (40/60, not 30/70) - ensures trades
3. 1w HTF HMA for bias only (not multiple HTFs)
4. ATR(14) trailing stoploss at 2.5x - mandatory risk management
5. Discrete sizing: 0.0, ±0.30 - minimizes fee churn
6. NO Choppiness, NO Donchian, NO session filters - these killed trades

Why this should work on 1d:
- 1d naturally produces 20-50 trades/year (perfect for fee management)
- HMA crossover is proven across multiple winning strategies
- RSI 40/60 thresholds ensure entries happen (not waiting for extremes)
- Single HTF (1w) for bias, not 3+ HTFs that conflict
- Target: 40-80 trades on train (2021-2024), 10-20 on test (2025-2026)

Target: Sharpe > 0.612, 40-80 trades/train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_simple_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    hma_21 = calculate_hma(close, 21)
    hma_63 = calculate_hma(close, 63)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    atr_median = np.nanmedian(atr_14[100:])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 1d (target 20-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_63[i]):
            continue
        
        # === HTF BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA crossover) ===
        hma_bullish = hma_21[i] > hma_63[i]
        hma_bearish = hma_21[i] < hma_63[i]
        
        # === RSI WITH MODERATE THRESHOLDS (ensures trades happen) ===
        # Using 40/60 instead of 30/70 - less extreme, more trades
        rsi_long = rsi_14[i] < 45.0  # Pullback in uptrend
        rsi_short = rsi_14[i] > 55.0  # Rally in downtrend
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5  # Reduce to 50% in extreme vol
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.75  # Reduce to 75%
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - Simpler confluence (HTF bias + trend + RSI pullback)
        if price_above_hma_1w and hma_bullish:
            # Primary: HTF bullish + trend bullish + RSI pullback
            if rsi_long:
                desired_signal = position_size
            # Secondary: Strong trend (HMA spread wide) - enter on any RSI < 55
            elif (hma_21[i] - hma_63[i]) / close[i] > 0.02 and rsi_14[i] < 55.0:
                desired_signal = position_size
        
        # SHORT SETUP - Simpler confluence
        if price_below_hma_1w and hma_bearish:
            # Primary: HTF bearish + trend bearish + RSI rally
            if rsi_short:
                desired_signal = -position_size
            # Secondary: Strong trend (HMA spread wide) - enter on any RSI > 45
            elif (hma_63[i] - hma_21[i]) / close[i] > 0.02 and rsi_14[i] > 45.0:
                desired_signal = -position_size
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === RSI EXIT (extreme reached - take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 65.0:
            # Long exit when RSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 35.0:
            # Short exit when RSI reaches oversold
            desired_signal = 0.0
        
        # === TREND EXIT (HMA crossover reversal) ===
        if in_position and position_side > 0 and hma_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish:
            desired_signal = 0.0
        
        # === HTF BIAS EXIT ===
        if in_position and position_side > 0 and price_below_hma_1w:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and hma_bullish and price_above_hma_1w:
                desired_signal = position_size
            elif position_side < 0 and hma_bearish and price_below_hma_1w:
                desired_signal = -position_size
        
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