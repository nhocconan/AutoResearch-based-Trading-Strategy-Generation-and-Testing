#!/usr/bin/env python3
"""
Experiment #549: 15m Primary + 1h/1d HTF — Simplified RSI Pullback with Session Filter

Hypothesis: 15m timeframe needs LOOSER entry conditions to generate trades (previous 15m 
strategies all returned 0 trades = auto-reject). Using 1d/1h HTF for direction bias, 
15m RSI(7) for sensitive entry timing, and session filter to reduce noise.

Key learnings from failed 15m experiments (#537, #541, #545):
1. Entry conditions were TOO STRICT → 0 trades generated
2. Need RSI(7) not RSI(14) for more sensitive entries on lower TF
3. HTF alignment should be OR not AND (price > 1d_HMA OR 1h_HMA = bullish)
4. Session filter helps but shouldn't block all entries
5. Size must be smaller (0.15-0.20) due to higher frequency

Strategy logic:
1. 1d HMA(21) = macro bias (price > HMA = long bias)
2. 1h HMA(21) = medium bias (confirms 1d direction)
3. 15m RSI(7) = entry trigger (oversold <30, overbought >70)
4. 15m HMA(9) = momentum confirmation (price > HMA = bullish momentum)
5. Session filter: prefer 00-12 UTC (London+NY overlap)
6. ATR(14)*2.0 stoploss on all positions

Entry conditions (LOOSE to ensure trades):
- Long: 1d_HMA bullish OR 1h_HMA bullish + RSI(7)<35 + price>15m_HMA
- Short: 1d_HMA bearish OR 1h_HMA bearish + RSI(7)>65 + price<15m_HMA

Target: 50-100 trades/year, Sharpe>0.40, DD<-30%
Timeframe: 15m
Size: 0.15-0.20 (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_hma_1h1d_session_v3"
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

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hour = (open_time // 3600000) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1h HMA for medium trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=9)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
    SIZE_STRONG = 0.22
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d macro + 1h medium) ===
        # Use OR logic for looser entries (either HTF confirms direction)
        htf_bull = (close[i] > hma_1d_aligned[i]) or (close[i] > hma_1h_aligned[i])
        htf_bear = (close[i] < hma_1d_aligned[i]) or (close[i] < hma_1h_aligned[i])
        
        # Strong bias when both HTF agree
        htf_strong_bull = (close[i] > hma_1d_aligned[i]) and (close[i] > hma_1h_aligned[i])
        htf_strong_bear = (close[i] < hma_1d_aligned[i]) and (close[i] < hma_1h_aligned[i])
        
        # === 15m MOMENTUM ===
        momentum_bull = close[i] > hma_15m[i]
        momentum_bear = close[i] < hma_15m[i]
        
        # HMA slope (5-bar lookback)
        hma_slope_bull = hma_15m[i] > hma_15m[i-5] if i >= 5 and not np.isnan(hma_15m[i-5]) else False
        hma_slope_bear = hma_15m[i] < hma_15m[i-5] if i >= 5 and not np.isnan(hma_15m[i-5]) else False
        
        # === RSI EXTREMES (RSI(7) for sensitivity) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_oversold = rsi_7[i] < 25.0
        rsi_extreme_overbought = rsi_7[i] > 75.0
        
        # RSI recovery signals
        rsi_rising = rsi_7[i] > rsi_7[i-1] if i > 0 else False
        rsi_falling = rsi_7[i] < rsi_7[i-1] if i > 0 else False
        
        # === SESSION FILTER (prefer 00-12 UTC) ===
        hour = get_session_hour(open_time[i])
        prime_session = (hour >= 0 and hour <= 12)  # London + NY overlap
        
        # === ENTRY LOGIC (LOOSE to ensure trades) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_bull and momentum_bull:
            # Primary: RSI oversold + momentum up
            if rsi_oversold and hma_slope_bull:
                desired_signal = SIZE_STRONG if htf_strong_bull else SIZE_BASE
            # Secondary: RSI recovery from oversold
            elif rsi_7[i] < 40.0 and rsi_rising and momentum_bull:
                desired_signal = SIZE_BASE * 0.9
            # Tertiary: Simple momentum continuation
            elif htf_strong_bull and hma_slope_bull and rsi_7[i] > 40.0 and rsi_7[i] < 60.0:
                desired_signal = SIZE_BASE * 0.8
        
        # SHORT entries
        elif htf_bear and momentum_bear:
            # Primary: RSI overbought + momentum down
            if rsi_overbought and hma_slope_bear:
                desired_signal = -SIZE_STRONG if htf_strong_bear else -SIZE_BASE
            # Secondary: RSI recovery from overbought
            elif rsi_7[i] > 60.0 and rsi_falling and momentum_bear:
                desired_signal = -SIZE_BASE * 0.9
            # Tertiary: Simple momentum continuation
            elif htf_strong_bear and hma_slope_bear and rsi_7[i] > 40.0 and rsi_7[i] < 60.0:
                desired_signal = -SIZE_BASE * 0.8
        
        # Session filter: reduce size outside prime hours (don't block entries)
        if not prime_session and desired_signal != 0.0:
            desired_signal = desired_signal * 0.7
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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