#!/usr/bin/env python3
"""
Experiment #131: 4h Primary + 1d/1w HTF — Volatility Squeeze Expansion with KAMA Trend

Hypothesis: Previous Donchian breakout strategies failed because breakouts in low-vol
environments often reverse. This uses Bollinger Band Width to detect volatility squeezes
(contraction), then enters on expansion ONLY when HTF trend aligns. Key differences:

1) KAMA (Kaufman Adaptive MA) instead of HMA — adapts to market noise, fewer whipsaws
2) BB Width percentile for squeeze detection — not Choppiness Index (failed in #119-#127)
3) Volatility expansion trigger — BB Width increases >20% from squeeze low
4) 1d KAMA for directional bias — only trade expansion in HTF trend direction
5) 1w HMA for macro filter — skip trades against weekly trend

Why this should work:
- Volatility squeezes precede major moves (Bollinger Squeeze theory)
- KAMA filters noise better than EMA/HMA in ranging markets (2025 bear)
- Expansion confirmation reduces false breakouts
- 4h timeframe = 20-50 trades/year target (low fee drag)
- Simpler than dual-regime (which failed in #119, #121, #122)

Position size: 0.25 base, 0.30 max with HTF confluence
Stoploss: 2.5*ATR trailing
Target: Sharpe > 0.5 on ALL symbols, 25-40 trades/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_bb squeeze_expansion_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = (close_s - close_s.shift(period)).abs()
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    er = change / (volatility + 1e-10)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = pd.Series(index=close_s.index, dtype=float)
    kama.iloc[period-1] = close_s.iloc[period-1]
    for i in range(period, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    return kama.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma * 100.0
    return upper.values, lower.values, bandwidth.values, sma.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate BB Width percentile rank over lookback period."""
    bb_pct = np.zeros(len(bandwidth))
    for i in range(lookback, len(bandwidth)):
        window = bandwidth[i-lookback:i+1]
        rank = np.sum(window < bandwidth[i]) / len(window)
        bb_pct[i] = rank * 100.0
    return bb_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d KAMA for trend bias
    kama_1d = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1d KAMA slope
    kama_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(kama_1d_aligned[i]) and not np.isnan(kama_1d_aligned[i-1]) and kama_1d_aligned[i-1] != 0:
            kama_1d_slope[i] = (kama_1d_aligned[i] - kama_1d_aligned[i-1]) / kama_1d_aligned[i-1] * 100
        else:
            kama_1d_slope[i] = 0.0
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    bb_percentile = calculate_bb_percentile(bb_width, lookback=100)
    kama_4h_10 = calculate_kama(close, period=10)
    kama_4h_30 = calculate_kama(close, period=30)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # Track squeeze state
    squeeze_detected = False
    squeeze_low = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(bb_width[i]) or np.isnan(bb_percentile[i]):
            continue
        if np.isnan(kama_4h_10[i]) or np.isnan(kama_4h_30[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === HTF TREND BIAS ===
        # 1d KAMA direction
        kama_1d_bullish = kama_1d_slope[i] > 0.3
        kama_1d_bearish = kama_1d_slope[i] < -0.3
        kama_1d_flat = abs(kama_1d_slope[i]) <= 0.3
        
        # Price vs 1d KAMA
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # 1w HMA macro trend
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY SQUEEZE DETECTION ===
        # Squeeze: BB Width percentile < 20 (bottom 20% of recent range)
        squeeze_active = bb_percentile[i] < 20.0
        
        # Track squeeze low for expansion detection
        if squeeze_active:
            if not squeeze_detected or bb_width[i] < squeeze_low:
                squeeze_detected = True
                squeeze_low = bb_width[i]
        
        # === VOLATILITY EXPANSION TRIGGER ===
        # Expansion: BB Width increases >20% from squeeze low
        expansion_long = False
        expansion_short = False
        
        if squeeze_detected and squeeze_low > 0:
            expansion_threshold = squeeze_low * 1.20
            if bb_width[i] > expansion_threshold:
                # Direction determined by price break of BB bands
                expansion_long = close[i] > bb_upper[i]
                expansion_short = close[i] < bb_lower[i]
        
        # === 4h TREND FILTER ===
        kama_4h_bullish = kama_4h_10[i] > kama_4h_30[i]
        kama_4h_bearish = kama_4h_10[i] < kama_4h_30[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 1w trend up + 1d trend up + squeeze expansion + 4h trend up
        if price_above_hma_1w and (kama_1d_bullish or price_above_kama_1d):
            if expansion_long and kama_4h_bullish:
                # Check confluence for position size
                if kama_1d_bullish and price_above_hma_1w:
                    new_signal = POSITION_SIZE_MAX
                else:
                    new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        # Require: 1w trend down + 1d trend down + squeeze expansion + 4h trend down
        if price_below_hma_1w and (kama_1d_bearish or price_below_kama_1d):
            if expansion_short and kama_4h_bearish:
                if kama_1d_bearish and price_below_hma_1w:
                    new_signal = -POSITION_SIZE_MAX
                else:
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if still in expansion phase and HTF trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price > BB mid and 1d trend intact
                if close[i] > bb_mid[i] and price_above_kama_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price < BB mid and 1d trend intact
                if close[i] < bb_mid[i] and price_below_kama_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
            squeeze_detected = False  # Reset squeeze on stop
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_1w or (kama_1d_bearish and price_below_kama_1d):
                new_signal = 0.0
                squeeze_detected = False
        
        if in_position and position_side < 0:
            if price_above_hma_1w or (kama_1d_bullish and price_above_kama_1d):
                new_signal = 0.0
                squeeze_detected = False
        
        # === EXIT ON OPPOSITE EXPANSION ===
        if in_position and position_side > 0 and expansion_short:
            new_signal = 0.0
            squeeze_detected = False
        
        if in_position and position_side < 0 and expansion_long:
            new_signal = 0.0
            squeeze_detected = False
        
        # === RESET SQUEEZE IF BB WIDTH CONTRACTS AGAIN ===
        if squeeze_detected and bb_width[i] < squeeze_low * 1.05:
            squeeze_detected = False
        
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