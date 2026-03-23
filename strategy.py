#!/usr/bin/env python3
"""
Experiment #475: 1h Primary + 4h HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous 1h strategies (#465, #470) failed with 0 trades due to overly
strict entry conditions (session filters + volume + multiple confluence). This 
strategy uses MINIMAL filters to ensure trade frequency while maintaining quality:

1. 4h HMA(21) for trend direction (proven in best strategy mtf_4h_triple_regime)
2. 1h RSI(14) for pullback entries (oversold < 35 in uptrend, overbought > 65 in downtrend)
3. ATR(14) 2.5x trailing stoploss
4. NO session filter, NO volume filter, NO 1d filter (these killed trade frequency)
5. Discrete sizing: 0.0, ±0.25, ±0.30
6. Hold logic: maintain position while trend intact (reduces churn)

Why this should work: Simpler = more trades. RSI pullback in direction of HTF trend
is a proven pattern across all market regimes. 1h TF with 4h trend filter should
generate ~40-60 trades/year. Looser RSI thresholds (35/65 vs 15/85) ensure entries
in both strong and moderate pullbacks.

Target: Sharpe > 0.612, DD < -40%, trades >= 30 train, >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1h indicators (primary timeframe)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
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
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        # === TREND DIRECTION (4h HMA) ===
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # === RSI PULLBACK SIGNALS (looser thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI oversold (pullback in uptrend)
        if trend_bullish and rsi_oversold:
            desired_signal = SIZE_LONG
        
        # SHORT: 4h bearish + RSI overbought (pullback in downtrend)
        if desired_signal == 0.0 and trend_bearish and rsi_overbought:
            desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and trend_bullish:
                desired_signal = SIZE_LONG
            elif position_side < 0 and trend_bearish:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
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