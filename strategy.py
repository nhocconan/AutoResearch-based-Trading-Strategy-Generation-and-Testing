#!/usr/bin/env python3
"""
Experiment #040: 1h Primary + 4h/12h HTF — Simplified Trend Pullback with Relaxed Entries

Hypothesis: Previous 1h/30m strategies failed with Sharpe=0.000 (ZERO TRADES) due to 
overly strict confluence filters. This strategy MAXIMIZES trade generation while 
maintaining HTF trend filter:

CRITICAL LESSON FROM 36 FAILED STRATEGIES:
- RSI thresholds 30/70 or 15/85 = too strict = 0 trades
- CHOP filter = kills signal frequency
- Session/volume filters = compound to zero signals
- Solution: LOOSEN thresholds, remove kill-filters

Key changes:
1. RSI(14) thresholds 25/75 (wider than 30/70) - ensures trades trigger
2. NO CHOP filter - was killing trade frequency in 5+ failed strategies
3. NO session filter - crypto trades 24/7, this was artificial constraint
4. Volume filter relaxed to 0.5x (not 1.5x) - just avoid dead zones
5. Entry on RSI LEVEL (not cross) - more frequent triggers
6. Size 0.25 with discrete levels

Entry Logic (SIMPLIFIED):
- LONG: 4h HMA bullish + 12h HMA bullish + RSI(14) < 30 (oversold pullback in uptrend)
- SHORT: 4h HMA bearish + 12h HMA bearish + RSI(14) > 70 (overbought pullback in downtrend)
- Exit: RSI crosses 50 (momentum fade) OR stoploss hit

Risk: 2.5x ATR trailing stop, position tracking
Target: 50-100 trades/year, Sharpe>0.3, DD>-40%, ALL symbols must trade
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h12h_hma_relaxed_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend filter"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
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
    
    rsi = np.full(n, np.nan)
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
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for major trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for minimal filter (just avoid dead zones)
    vol_sma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # RSI exit tracking
    rsi_entry_value = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h and 12h) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === VOLUME FILTER (minimal - just avoid dead zones) ===
        vol_ok = True
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 0:
            vol_ok = volume[i] > 0.5 * vol_sma[i]
        
        # === RSI ENTRY (LOOSE thresholds for trade generation) ===
        # Long: RSI < 30 (oversold) in uptrend
        rsi_oversold = rsi[i] < 30.0
        # Short: RSI > 70 (overbought) in downtrend
        rsi_overbought = rsi[i] > 70.0
        
        # === RSI EXIT (momentum fade) ===
        rsi_exit_long = False
        rsi_exit_short = False
        if in_position:
            # Exit long when RSI recovers to 55+
            if position_side > 0 and rsi[i] > 55.0:
                rsi_exit_long = True
            # Exit short when RSI drops to 45-
            if position_side < 0 and rsi[i] < 45.0:
                rsi_exit_short = True
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: Both HTF bullish + RSI oversold + volume ok
        if hma_4h_bull and hma_12h_bull and rsi_oversold and vol_ok:
            desired_signal = BASE_SIZE
        
        # SHORT: Both HTF bearish + RSI overbought + volume ok
        elif hma_4h_bear and hma_12h_bear and rsi_overbought and vol_ok:
            desired_signal = -BASE_SIZE
        
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
        
        # === RSI EXIT OVERRIDE ===
        if in_position and ((position_side > 0 and rsi_exit_long) or (position_side < 0 and rsi_exit_short)):
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
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
                rsi_entry_value = rsi[i]
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                rsi_entry_value = rsi[i]
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
                rsi_entry_value = 0.0
        
        signals[i] = final_signal
    
    return signals