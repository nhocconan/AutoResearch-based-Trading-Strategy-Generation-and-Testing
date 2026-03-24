#!/usr/bin/env python3
"""
Experiment #129: 15m Primary + 1h/4h HTF — RSI Mean Reversion + HMA Trend + Session Filter

Hypothesis: 15m strategies failed (Sharpe=0.000) because entry conditions were TOO STRICT.
This version uses LOOSE RSI(7) extremes (30/70 not 20/80) + Bollinger band proximity + 
session filter (00-12 UTC) to ensure trades actually generate while maintaining selectivity.

Key design choices:
- Timeframe: 15m (target 40-100 trades/year)
- HTF: 1h HMA(21) for trend bias, 4h HMA(50) for regime
- Entry: RSI(7) < 30 long, > 70 short + price near BB(20,2.0) bands
- Session: 00-12 UTC only (London/NY overlap, 50% of bars eligible)
- Position size: 0.20 (20% of capital, smaller for 15m frequency)
- Stoploss: 2.5x ATR trailing
- LOOSE filters to ensure >=10 trades on train, >=3 on test per symbol

Why this should work on 15m:
- RSI(7) extremes happen frequently enough on 15m
- BB proximity adds confluence without being too restrictive
- HTF bias prevents counter-trend trades in strong trends
- Session filter reduces noise during low-volume hours
- Smaller size (0.20) accounts for higher trade frequency

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=10 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi7_bb_hma_session_1h4h_v1"
timeframe = "15m"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def get_hour_from_open_time(open_time_array):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds, convert to hours
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMA for trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (15m) indicators
    rsi = calculate_rsi(close, period=7)  # Fast RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    hma_15m = calculate_hma(close, period=21)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (smaller for 15m frequency)
    
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
        if np.isnan(rsi[i]) or np.isnan(hma_15m[i]):
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
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London/NY overlap) ===
        in_session = hours[i] < 12  # 00:00 to 11:59 UTC
        
        # === HTF BIAS (1h and 4h HMA) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # Strong HTF bias when both agree
        htf_strong_bull = htf_1h_bull and htf_4h_bull
        htf_strong_bear = htf_1h_bear and htf_4h_bear
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI EXTREMES (LOOSE for 15m - ensure trades) ===
        rsi_oversold = rsi[i] < 30.0  # Long signal
        rsi_overbought = rsi[i] > 70.0  # Short signal
        
        # === BOLLINGER BAND PROXIMITY ===
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range > 1e-10:
            bb_position = (close[i] - bb_lower[i]) / bb_range
            near_lower = bb_position < 0.15  # Within 15% of lower band
            near_upper = bb_position > 0.85  # Within 15% of upper band
        else:
            near_lower = False
            near_upper = False
        
        # === DESIRED SIGNAL (Mean Reversion with HTF Filter) ===
        desired_signal = 0.0
        
        # LONG: RSI oversold + near BB lower + HTF not strongly bear
        if rsi_oversold and near_lower:
            if not htf_strong_bear:  # Allow long if HTF neutral or bull
                if in_session:
                    desired_signal = SIZE
                else:
                    desired_signal = SIZE * 0.5  # Reduced size outside session
        
        # SHORT: RSI overbought + near BB upper + HTF not strongly bull
        elif rsi_overbought and near_upper:
            if not htf_strong_bull:  # Allow short if HTF neutral or bear
                if in_session:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.5  # Reduced size outside session
        
        # Fallback: RSI extreme alone (very loose, ensures trades)
        elif rsi[i] < 25.0 and hma_bull:
            desired_signal = SIZE * 0.5
        elif rsi[i] > 75.0 and hma_bear:
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.4:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.4:
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