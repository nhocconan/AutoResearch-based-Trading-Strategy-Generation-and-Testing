#!/usr/bin/env python3
"""
Experiment #233: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe requires EXTREME selectivity to avoid fee drag. Previous 
experiments failed because they either traded too frequently (>200/year) or had 
0 trades (overly strict filters). This strategy uses:

1. 4h HMA(21) for MAJOR trend bias - only trade in HTF direction
2. 15m RSI(14) for intermediate momentum confirmation
3. 5m RSI(3) for precise pullback entry timing
4. SESSION FILTER: Only 08:00-20:00 UTC (high liquidity, avoid Asia low-vol)
5. Volume filter: Current volume > 1.5x 20-bar average
6. ATR(14) trailing stoploss at 2.5x

Key insight from research: 5m strategies fail due to fee drag from overtrading.
By requiring 4h trend + 15m momentum + 5m entry + session + volume = 5 confluence
factors, we limit to ~60-100 trades/year while maintaining edge.

Position sizing: 0.15 base (smaller due to 5m frequency), 0.25 for strong signals
Stoploss: 2.5x ATR trailing - CRITICAL for 5m volatility

Target: Sharpe>0.40 (beat current best 0.399), DD>-30%, trades>=50 train, trades>=5 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_pullback_hma_15m4h_v1"
timeframe = "5m"
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

def is_session_active(open_time_ms, start_hour=8, end_hour=20):
    """
    Check if timestamp is within active session (08:00-20:00 UTC)
    open_time_ms: milliseconds since epoch (Binance format)
    """
    # Convert ms to datetime
    ts_ms = open_time_ms
    ts_sec = ts_ms / 1000.0
    # Get hour in UTC
    hour = int((ts_sec % 86400) / 3600)
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for major trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 15m RSI for intermediate momentum
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate primary (5m) indicators
    rsi_5m_short = calculate_rsi(close, period=3)
    rsi_5m_std = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_5m = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15  # 15% base position size (smaller for 5m frequency)
    SIZE_STRONG = 0.25  # 25% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_5m_short[i]) or np.isnan(rsi_5m_std[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08:00-20:00 UTC only) ===
        in_session = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === VOLUME FILTER (current > 1.5x 20-bar avg) ===
        vol_filter = volume[i] > 1.5 * vol_sma[i]
        
        # === 4h HTF TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m INTERMEDIATE MOMENTUM ===
        rsi_15m_bull = rsi_15m_aligned[i] > 50.0
        rsi_15m_bear = rsi_15m_aligned[i] < 50.0
        rsi_15m_strong_bull = rsi_15m_aligned[i] > 55.0
        rsi_15m_strong_bear = rsi_15m_aligned[i] < 45.0
        
        # === 5m ENTRY TIMING (RSI pullback in trend) ===
        rsi_5m_oversold = rsi_5m_short[i] < 25.0
        rsi_5m_overbought = rsi_5m_short[i] > 75.0
        rsi_5m_neutral = 35.0 < rsi_5m_short[i] < 65.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === 5m HMA TREND ===
        hma_5m_bull = close[i] > hma_5m[i]
        hma_5m_bear = close[i] < hma_5m[i]
        
        # === ENTRY LOGIC (require 4+ confluence factors) ===
        desired_signal = 0.0
        
        # LONG setup: 4h bull + 15m bull + 5m pullback + session + volume
        if htf_4h_bull and rsi_15m_bull and in_session and vol_filter:
            confluence_count = 0
            if rsi_15m_strong_bull:
                confluence_count += 1
            if rsi_5m_oversold or rsi_5m_neutral:
                confluence_count += 1
            if hma_5m_bull:
                confluence_count += 1
            if above_sma200:
                confluence_count += 1
            
            # Require 3+ confluence for entry
            if confluence_count >= 3:
                if rsi_5m_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT setup: 4h bear + 15m bear + 5m pullback + session + volume
        elif htf_4h_bear and rsi_15m_bear and in_session and vol_filter:
            confluence_count = 0
            if rsi_15m_strong_bear:
                confluence_count += 1
            if rsi_5m_overbought or rsi_5m_neutral:
                confluence_count += 1
            if hma_5m_bear:
                confluence_count += 1
            if below_sma200:
                confluence_count += 1
            
            # Require 3+ confluence for entry
            if confluence_count >= 3:
                if rsi_5m_overbought:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
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