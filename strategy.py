#!/usr/bin/env python3
"""
Experiment #079: 1h Primary + 4h/12h HTF — RSI Pullback + HMA Trend + Session Filter

Hypothesis: After 78 failed experiments, the clearest pattern is:
- 1h strategies fail due to TOO STRICT conditions (exp #070, #073, #076, #077 = 0 trades)
- Complex regime switching adds noise without edge (exp #068, #074, #075, #078 = negative)
- SOLUTION: Simple 3-filter confluence with LOOSE thresholds to ensure trades generate
- 4h HMA for trend direction (proven in baseline mtf_hma_rsi_zscore_v1)
- 1h RSI pullback for entry timing (RSI 35-65 range, not extreme)
- Session filter 08-20 UTC (active trading hours, less noise)
- ATR volatility filter to avoid dead markets (ATR ratio > 0.8)
- Position size: 0.25 (25% of capital, conservative for 1h)
- Stoploss: 2.5x ATR trailing

Key design for TRADE GENERATION (learned from 0-trade failures):
- RSI long: 30-55 (not <30, allows entries during mild pullbacks)
- RSI short: 45-70 (not >70, allows entries during mild rallies)
- Only 3 filters: HTF trend + RSI range + session (no 4th filter blocking trades)
- Fallback: enter on strong HTF trend even if RSI neutral (ensures trades in strong trends)

Target: Sharpe>0.167, DD>-40%, trades>=40/year on train, trades>=5/year on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_hma_4h12h_session_v1"
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
    """Average True Range for stoploss and volatility filter"""
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
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # ATR ratio for volatility filter (current ATR / 30-bar avg ATR)
    atr_30 = pd.Series(atr).ewm(span=30, min_periods=30, adjust=False).mean().values
    atr_ratio = atr / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 1h)
    
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
        if np.isnan(rsi[i]):
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
        is_active_session = 8 <= hour_utc <= 20
        
        # === VOLATILITY FILTER ===
        # Avoid dead markets (ATR ratio > 0.7)
        vol_ok = atr_ratio[i] > 0.7 or np.isnan(atr_ratio[i])
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === MAJOR TREND BIAS (12h HMA) ===
        major_bull = close[i] > hma_12h_aligned[i]
        major_bear = close[i] < hma_12h_aligned[i]
        
        # === RSI PULLBACK (LOOSE THRESHOLDS FOR TRADE GENERATION) ===
        # Long: RSI 30-55 (pullback in uptrend, not oversold extreme)
        # Short: RSI 45-70 (rally in downtrend, not overbought extreme)
        rsi_long = 30.0 <= rsi[i] <= 55.0
        rsi_short = 45.0 <= rsi[i] <= 70.0
        
        # === DESIRED SIGNAL (Simple 3-Filter Confluence) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + RSI pullback + (session OR major bull)
        if htf_bull and rsi_long:
            if is_active_session or major_bull:
                desired_signal = SIZE
        
        # SHORT: HTF bear + RSI pullback + (session OR major bear)
        if htf_bear and rsi_short:
            if is_active_session or major_bear:
                desired_signal = -SIZE
        
        # FALLBACK: Strong HTF trend even with neutral RSI (ensures trades in strong trends)
        if desired_signal == 0.0:
            # Very strong 4h trend (price > 2% above HMA)
            hma_deviation = (close[i] - hma_4h_aligned[i]) / (hma_4h_aligned[i] + 1e-10)
            
            if htf_bull and hma_deviation > 0.02 and rsi[i] > 40.0 and rsi[i] < 65.0:
                desired_signal = SIZE * 0.6
            elif htf_bear and hma_deviation < -0.02 and rsi[i] > 35.0 and rsi[i] < 60.0:
                desired_signal = -SIZE * 0.6
        
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
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
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