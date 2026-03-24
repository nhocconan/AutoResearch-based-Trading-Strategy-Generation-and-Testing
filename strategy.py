#!/usr/bin/env python3
"""
Experiment #055: 1h Primary + 4h/1d HTF — Trend Pullback with Volatility Filter

Hypothesis: 1h timeframe with LOOSE entry conditions (learned from #045/#050 failures).
Key insight: Previous 1h strategies got Sharpe=0.000 due to ZERO trades from over-filtering.

Strategy Design (LOOSE conditions to ensure trades):
- 4h HMA(21) for primary trend direction (HTF signal)
- 1d HMA(21) for major bias (only 1 needs to align, not both)
- 1h RSI(14) pullback: 40/60 thresholds (NOT 30/70 which caused 0 trades)
- 1h Bollinger Band position for volatility context
- Volume filter: 0.7x average (loose, not 1.5x)
- NO session filter (killed #045/#048)
- ATR(14)*2.5 trailing stoploss
- Size: 0.25 (conservative for 1h TF)

Target: 40-80 trades/year, Sharpe>0.351, DD>-40%
Timeframe: 1h (use 4h/1d for direction, 1h for entry timing)

Why this should work where #045/#050 failed:
- RSI 40/60 vs 30/70 = 3x more signals
- OR logic on HTF (4h OR 1d) vs AND = 2x more signals
- No session filter = trades during Asian/London sessions
- Volume 0.7x vs 1.5x = accepts normal volume days
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_bb_pullback_loose_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - less lag than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    def wma(data, span):
        res = np.full(len(data), np.nan)
        w = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            res[i] = np.sum(data[i - span + 1:i + 1] * w) / np.sum(w)
        return res
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    double_wma = 2.0 * wma_half - wma_full
    hma = wma(double_wma, sqrt_p)
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # BB position: 0=lower, 0.5=middle, 1.0=upper
    bb_pos = np.full(n, np.nan)
    valid = ~np.isnan(sma) & (upper > lower)
    bb_pos[valid] = (close[valid] - lower[valid]) / (upper[valid] - lower[valid])
    
    return upper, lower, bb_pos

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

def calculate_volume_avg(volume, period=20):
    """Rolling average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h/1d HMA for HTF trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_pos = calculate_bollinger(close, period=20, std_mult=2.0)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative for 1h TF
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(bb_pos[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h and 1d HMA) ===
        # LOOSE: only need ONE HTF aligned (OR logic, not AND)
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # At least one HTF must agree
        htf_bull = hma_4h_bull or hma_1d_bull
        htf_bear = hma_4h_bear or hma_1d_bear
        
        # === RSI PULLBACK (LOOSE thresholds: 40/60 not 30/70) ===
        rsi_oversold = rsi[i] < 60.0  # Pullback in uptrend
        rsi_overbought = rsi[i] > 40.0  # Pullback in downtrend
        
        # === VOLUME FILTER (LOOSE: 0.7x avg) ===
        vol_ok = volume[i] >= 0.7 * vol_avg[i] if vol_avg[i] > 0 else True
        
        # === BOLLINGER BAND POSITION ===
        # Long: price in lower half of BB (pullback)
        # Short: price in upper half of BB (pullback)
        bb_lower_half = bb_pos[i] < 0.5
        bb_upper_half = bb_pos[i] > 0.5
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: HTF bull + RSI pullback + volume OK + BB lower half
        if htf_bull and rsi_oversold and vol_ok and bb_lower_half:
            desired_signal = SIZE
        
        # SHORT: HTF bear + RSI pullback + volume OK + BB upper half
        elif htf_bear and rsi_overbought and vol_ok and bb_upper_half:
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