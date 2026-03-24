#!/usr/bin/env python3
"""
Experiment #617: 15m Primary + 4h/12h HTF — Simple Trend Pullback with RSI(7)

Hypothesis: 15m timeframe has been unexplored (0 experiments). Key insight from rules:
"For 15m/30m/1h: use HTF (4h/12h) for signal DIRECTION, lower TF only for entry TIMING."

This strategy uses:
1. 4h HMA(21) = primary trend direction (LONG vs SHORT bias)
2. 12h HMA(21) = macro confirmation (avoid counter-trend trades)
3. 15m RSI(7) = pullback entry timing (faster than RSI14, catches shallow pullbacks)
4. 15m HMA(9) = local momentum confirmation
5. Session preference (00-12 UTC) but NOT required - avoids 0 trades problem
6. ATR(14)*2.5 stoploss on all positions

Key differences from failed 15m experiments (#605, #609, #613):
- SIMPLER entry logic (3 filters max, not 5+)
- RSI(7) with wider thresholds (35/65 not 25/75) - generates more trades
- NO ADX filter (too restrictive, causes 0 trades)
- Session is preference not requirement
- Size 0.20-0.25 (smaller for 15m frequency)

Target: 50-100 trades/year, Sharpe>0.40, DD<-30%
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_4h12h_simple_v1"
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
    """Average True Range"""
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
    
    # Calculate and align 4h HMA for primary trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=9)
    rsi_15m = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr_15m = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_15m[i]) or atr_15m[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_15m[i]):
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
        
        # === HTF TREND BIAS (4h primary + 12h confirmation) ===
        # 4h HMA determines primary direction
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # 12h HMA for macro confirmation (avoid counter-trend)
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both HTF agree
        htf_strong_bull = htf_4h_bull and htf_12h_bull
        htf_strong_bear = htf_4h_bear and htf_12h_bear
        
        # === 15m LOCAL TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # HMA slope (5-bar lookback)
        hma_slope_bull = hma_15m[i] > hma_15m[i-5] if i >= 5 and not np.isnan(hma_15m[i-5]) else False
        hma_slope_bear = hma_15m[i] < hma_15m[i-5] if i >= 5 and not np.isnan(hma_15m[i-5]) else False
        
        # === RSI PULLBACK (wider thresholds for more trades) ===
        # For longs: RSI pulled back but not oversold (35-50 range)
        rsi_pullback_long = rsi_15m[i] >= 35.0 and rsi_15m[i] <= 50.0
        # For shorts: RSI pulled back but not overbought (50-65 range)
        rsi_pullback_short = rsi_15m[i] >= 50.0 and rsi_15m[i] <= 65.0
        
        # RSI momentum (rising for long, falling for short)
        rsi_rising = rsi_15m[i] > rsi_15m[i-1] if i > 0 else False
        rsi_falling = rsi_15m[i] < rsi_15m[i-1] if i > 0 else False
        
        # === SESSION FILTER (preference, not requirement) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        is_prime_session = 0 <= hour_utc <= 12  # London + NY overlap
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bull + RSI pullback + 15m HMA confirmation
        if htf_4h_bull:
            # Strong long: both HTF bull + RSI pullback + HMA bull
            if htf_strong_bull and rsi_pullback_long and hma_15m_bull:
                if is_prime_session:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Standard long: 4h bull + RSI pullback + RSI rising
            elif rsi_pullback_long and rsi_rising and hma_slope_bull:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: 4h bear + RSI pullback + 15m HMA confirmation
        elif htf_4h_bear:
            # Strong short: both HTF bear + RSI pullback + HMA bear
            if htf_strong_bear and rsi_pullback_short and hma_15m_bear:
                if is_prime_session:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Standard short: 4h bear + RSI pullback + RSI falling
            elif rsi_pullback_short and rsi_falling and hma_slope_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_15m[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals