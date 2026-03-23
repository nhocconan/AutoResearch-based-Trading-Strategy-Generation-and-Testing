#!/usr/bin/env python3
"""
Experiment #370: 1h Primary + 4h/12h HTF — Relaxed Pullback Strategy

Hypothesis: 1h timeframe with 4h/12h trend bias can capture intraday moves within
the higher timeframe trend. Previous 1h strategies failed due to:
- #360: 0 trades (conditions too strict)
- #365: Negative Sharpe (too many trades / wrong logic)

KEY FIX: Relaxed entry conditions to ensure 30-60 trades/year while using
HTF for direction only. Entry on 1h RSI pullbacks within 4h/12h trend.

Strategy Logic:
1. 12h HMA(21) = MACRO BIAS (hard filter)
2. 4h HMA(21) = INTERMEDIATE TREND (entry direction)
3. 1h RSI(14) pullback = ENTRY TIMING (45/55 thresholds, relaxed)
4. Volume filter (>0.7x avg) = CONFIRMATION
5. Session filter (8-20 UTC) = LIQUIDITY
6. ATR(14) 2.5x trailing stop = RISK

TARGET: 30-60 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_pullback_4h12h_hma_rsi_relaxed_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (target 30-60 trades/year)
    
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
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(sma_20[i]) or np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # Session filter (8-20 UTC) - only trade during liquid hours
        try:
            hour = pd.to_datetime(prices['open_time'].iloc[i], unit='ms').hour
            in_session = 8 <= hour <= 20
        except:
            in_session = True  # Default to trading if timestamp parsing fails
        
        # Volume filter (>0.7x average)
        volume_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === MACRO BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: 12h bullish + 4h bullish + RSI pullback + volume + session
        if price_above_hma_12h and price_above_hma_4h:
            if rsi_14[i] < 50 and volume_ok and in_session:
                desired_signal = BASE_SIZE
        
        # SHORT: 12h bearish + 4h bearish + RSI pullback + volume + session
        elif price_below_hma_12h and price_below_hma_4h:
            if rsi_14[i] > 50 and volume_ok and in_session:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and price_below_hma_4h:
            desired_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and price_above_hma_4h:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h still bullish
                if price_above_hma_4h:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h still bearish
                if price_below_hma_4h:
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