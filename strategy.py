#!/usr/bin/env python3
"""
Experiment #150: 1h Primary + 4h HTF — Simplified RSI Pullback Strategy

HYPOTHESIS: Recent 1h strategies (#140, #145, #148) generated ZERO trades due to 
overly strict confluence (session + volume + multiple indicators). This strategy 
uses SIMPLER, LOOSER entry conditions to ensure trades happen on ALL symbols.

KEY CHANGES FROM FAILED 1h STRATEGIES:
1. NO session filter (killed trade generation in #140, #145, #148)
2. MODERATE RSI thresholds (35/65 instead of 15/85) = more triggers
3. Relaxed volume filter (1.2x instead of 1.8x)
4. Single HTF filter (4h HMA only, not 4h+12h+1d)
5. Position size: 0.20-0.25 (conservative for 1h TF)

LOGIC:
- 4h HMA(21) for trend direction (load ONCE before loop via mtf_data)
- 1h RSI(14) pullback entries within HTF trend
- Long: 4h HMA bullish + RSI(14) pulls back to 35-45 + volume > 1.2x avg
- Short: 4h HMA bearish + RSI(14) rallies to 55-65 + volume > 1.2x avg
- Exit: RSI crosses 50 midpoint OR 2.0*ATR stoploss
- Target: 40-70 trades/year, Sharpe > 0.3 on ALL symbols

WHY THIS SHOULD WORK:
- Simpler = more trades (learned from 0-trade failures)
- RSI pullback in trend direction = proven edge (Kaufman, Connors)
- 4h trend filter reduces whipsaw vs pure 1h mean reversion
- Conservative sizing (0.20-0.25) controls drawdown in 2022 crash
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_hma_4h_simplified_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME FILTER (relaxed) ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 1.2
        
        # === ENTRY LOGIC (LOOSE thresholds for trade generation) ===
        new_signal = 0.0
        
        # Long: 4h bullish + RSI pullback to 35-45 zone + volume
        if price_above_hma_4h:
            if 35.0 <= rsi_14[i] <= 48.0:
                if volume_confirmed:
                    new_signal = POSITION_SIZE_BASE
                    if volume_ratio > 1.5:
                        new_signal = POSITION_SIZE_MAX
        
        # Short: 4h bearish + RSI rally to 55-65 zone + volume
        if price_below_hma_4h:
            if 52.0 <= rsi_14[i] <= 65.0:
                if volume_confirmed:
                    new_signal = -POSITION_SIZE_BASE
                    if volume_ratio > 1.5:
                        new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and no exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if RSI still below 55 (not overbought)
                if rsi_14[i] < 55.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if RSI still above 45 (not oversold)
                if rsi_14[i] > 45.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 65.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 35.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals