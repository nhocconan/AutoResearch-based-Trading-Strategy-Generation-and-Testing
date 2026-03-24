#!/usr/bin/env python3
"""
Experiment #249: 15m Primary + 1h/1d HTF — Session-Aware RSI Pullback v1

Hypothesis: 15m timeframe with strict session filtering + HTF trend alignment can capture
intraday momentum moves while avoiding fee drag from overtrading. Key design:

1. SESSION FILTER: Only trade 00-12 UTC (London/NY overlap = 80% of crypto volume)
   This alone reduces trades by ~60% while keeping highest-probability setups

2. HTF TREND BIAS: 1d HMA(50) for major direction, 1h HMA(21) for intermediate
   Only long when both bullish, only short when both bearish

3. 15m RSI(7) PULLBACK: Enter on RSI(7) < 35 in uptrend, > 65 in downtrend
   Faster RSI captures intraday reversals better than RSI(14)

4. VOLATILITY FILTER: ATR(14) must be > 0.5x its 50-bar average (avoid dead markets)

5. POSITION SIZING: 0.20 base (smaller for 15m frequency), discrete levels
   Stoploss: 2.0x ATR trailing

6. LOOSENED ENTRIES: Ensure 50-100 trades/year (previous 15m strategies got 0)
   - RSI thresholds: 35/65 (not 30/70)
   - Session: 00-12 UTC (not narrower)
   - HTF: only require 1d HMA (1h is optional confirmation)

Target: Sharpe>0.40 (beat current best 0.399), DD>-35%, trades>=50 train, trades>=5 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_pullback_hma_1h1d_v1"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_utc_hour(prices, idx):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    open_time = prices['open_time'].values[idx]
    # Convert ms to seconds, then to datetime
    ts_seconds = open_time / 1000.0
    utc_hour = (ts_seconds % 86400) / 3600.0
    return int(utc_hour)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr(high, low, close, period=50)
    hma_15m = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_15m[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        utc_hour = get_utc_hour(prices, i)
        in_session = (utc_hour >= 0 and utc_hour < 12)
        
        if not in_session:
            # Close existing positions outside session
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            continue
        
        # === VOLATILITY FILTER ===
        # Only trade when ATR is above 50% of its 50-bar average (avoid dead markets)
        vol_filter = True
        if not np.isnan(atr_50[i]) and atr_50[i] > 1e-10:
            if atr[i] < 0.5 * atr_50[i]:
                vol_filter = False
        
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # === HTF TREND BIAS ===
        # 1d HMA for major trend (REQUIRED)
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1h HMA for intermediate trend (OPTIONAL confirmation)
        htf_1h_bull = not np.isnan(hma_1h_aligned[i]) and close[i] > hma_1h_aligned[i]
        htf_1h_bear = not np.isnan(hma_1h_aligned[i]) and close[i] < hma_1h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === SMA200 FILTER ===
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI(7) < 35 (oversold pullback in uptrend)
        rsi_oversold = rsi_7[i] < 35.0
        # Short: RSI(7) > 65 (overbought pullback in downtrend)
        rsi_overbought = rsi_7[i] > 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + RSI oversold + (1h bullish OR 15m HMA bullish)
        if htf_1d_bull and rsi_oversold:
            # Require at least one of: 1h bull, 15m HMA bull, above SMA200
            if htf_1h_bull or hma_bull or above_sma200:
                # Strong signal if all HTF align
                if htf_1h_bull and hma_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + RSI overbought + (1h bearish OR 15m HMA bearish)
        elif htf_1d_bear and rsi_overbought:
            if htf_1h_bear or hma_bear or below_sma200:
                if htf_1h_bear and hma_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
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