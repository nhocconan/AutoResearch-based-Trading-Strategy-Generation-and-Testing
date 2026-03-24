#!/usr/bin/env python3
"""
Experiment #159: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Choppiness + Session

Hypothesis: 1h timeframe with strict HTF alignment can achieve 40-80 trades/year
while maintaining positive Sharpe. Key insight from failures: too many filters = 0 trades.

New approach (learned from 150+ failures):
- 4h HMA(21) for trend direction (HTF bias)
- 12h HMA(50) for major regime filter (only trade with major trend)
- RSI(14) pullback entries within trend (RSI 35-55 long, 45-65 short)
- Choppiness(14) < 60 to avoid chop (but not too strict)
- Session filter: 08-20 UTC (high volume hours only)
- ATR(14) 2.5x trailing stop for risk management
- Position size: 0.25 (25% of capital)

CRITICAL: Entry conditions must be LOOSE enough to generate trades.
RSI thresholds widened from previous failures that got 0 trades.
Fallback entries when HTF trend is very strong (ignore CHOP).

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
Trade frequency: 40-80/year on 1h timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_session_4h12h_v1"
timeframe = "1h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    We use CHOP < 60 as filter (trending market only)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for major regime
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
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
        if np.isnan(hma_1h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]):
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
        
        # === SESSION FILTER (08-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        session_ok = (hour_utc >= 8) and (hour_utc <= 20)
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === MAJOR REGIME (12h HMA) ===
        major_bull = close[i] > hma_12h_aligned[i]
        major_bear = close[i] < hma_12h_aligned[i]
        
        # === CHOPPINESS FILTER ===
        # Only trade when CHOP < 60 (trending market)
        chop_ok = chop[i] < 60.0
        
        # === 1h HMA TREND ===
        hma_bull = close[i] > hma_1h[i]
        hma_bear = close[i] < hma_1h[i]
        
        # === RSI PULLBACK (LOOSE thresholds to ensure trades) ===
        # Long: RSI pulled back to 35-55 in uptrend
        # Short: RSI pulled back to 45-65 in downtrend
        rsi_ok_long = (rsi[i] >= 35.0) and (rsi[i] <= 55.0)
        rsi_ok_short = (rsi[i] >= 45.0) and (rsi[i] <= 65.0)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY: All conditions aligned (full size)
        if htf_bull and major_bull and chop_ok and rsi_ok_long and session_ok:
            desired_signal = SIZE
        
        elif htf_bear and major_bear and chop_ok and rsi_ok_short and session_ok:
            desired_signal = -SIZE
        
        # FALLBACK: Strong HTF trend (ignore CHOP if HTF very strong) - 80% size
        elif htf_bull and major_bull and rsi_ok_long:
            desired_signal = SIZE * 0.8
        
        elif htf_bear and major_bear and rsi_ok_short:
            desired_signal = -SIZE * 0.8
        
        # SECONDARY FALLBACK: 1h trend aligned with HTF (50% size)
        elif htf_bull and hma_bull and rsi[i] >= 40.0 and rsi[i] <= 60.0:
            desired_signal = SIZE * 0.5
        
        elif htf_bear and hma_bear and rsi[i] >= 40.0 and rsi[i] <= 60.0:
            desired_signal = -SIZE * 0.5
        
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
        elif desired_signal >= SIZE * 0.6:
            final_signal = SIZE * 0.8
        elif desired_signal <= -SIZE * 0.6:
            final_signal = -SIZE * 0.8
        elif desired_signal >= SIZE * 0.3:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.3:
            final_signal = -SIZE * 0.5
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