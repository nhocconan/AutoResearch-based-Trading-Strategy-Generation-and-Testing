#!/usr/bin/env python3
"""
Experiment #449: 15m Primary + 1h/1d HTF — Session-Filtered Mean Reversion

Hypothesis: 15m timeframe is too noisy for pure trend following (see #437, #441, #445 failures).
Instead, use 15m for MEAN REVERSION entries but ONLY in direction of 1d trend.
Key innovations:
1. DAILY TREND BIAS: Only long when price > 1d HMA(21), only short when price < 1d HMA(21)
2. 1H RSI TIMING: Use 1h RSI(7) extremes for entry (oversold long, overbought short)
3. SESSION FILTER: Only trade 00-12 UTC (London+NY overlap = highest liquidity, lowest whipsaw)
4. VOLATILITY FILTER: Skip entries when 15m ATR(14) > 1.5x 15m ATR(50) (vol spike = wait)
5. TIGHT STOPS: 2.0x ATR stoploss to preserve capital during 2022-style crashes

Why this should work on 15m:
- 1d trend filter prevents counter-trend trades (main killer on lower TF)
- 1h RSI gives cleaner signals than 15m RSI (less noise)
- Session filter cuts 50% of trades but keeps highest-quality setups
- Small position size (0.15-0.20) accounts for higher frequency

Target: Sharpe>0.45, DD>-35%, trades>=60 train (15/year), trades>=10 test
Timeframe: 15m (FIRST 15m experiment with proper HTF alignment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_1h1d_trend_v1"
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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=7)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate primary (15m) indicators
    atr_15m = calculate_atr(high, low, close, period=14)
    atr_15m_slow = calculate_atr(high, low, close, period=50)
    rsi_15m = calculate_rsi(close, period=14)
    bb_upper, bb_lower, _ = calculate_bollinger(close, period=20, std_dev=2.0)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_15m[i]) or atr_15m[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_15m[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        # Extract hour from open_time (milliseconds timestamp)
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // (1000 * 60 * 60)) % 24
        
        in_session = (hour_utc >= 0 and hour_utc < 12)
        
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY FILTER (skip vol spikes) ===
        vol_ratio = atr_15m[i] / atr_15m_slow[i] if atr_15m_slow[i] > 1e-10 else 1.0
        vol_spike = vol_ratio > 1.5
        
        if vol_spike:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === DAILY TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 1H RSI EXTREMES (entry timing) ===
        rsi_1h_oversold = rsi_1h_aligned[i] < 35.0
        rsi_1h_overbought = rsi_1h_aligned[i] > 65.0
        
        # === 15m RSI CONFIRMATION ===
        rsi_15m_oversold = rsi_15m[i] < 40.0
        rsi_15m_overbought = rsi_15m[i] > 60.0
        
        # === BB TOUCH (mean reversion trigger) ===
        touch_lower = close[i] <= bb_lower[i] if not np.isnan(bb_lower[i]) else False
        touch_upper = close[i] >= bb_upper[i] if not np.isnan(bb_upper[i]) else False
        
        # === SMA FILTER (trend confirmation) ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: Daily bull + 1h RSI oversold + (15m RSI oversold OR BB touch)
        if daily_bull:
            confluence_count = 0
            if rsi_1h_oversold:
                confluence_count += 1
            if rsi_15m_oversold:
                confluence_count += 1
            if touch_lower:
                confluence_count += 1
            if above_sma50:
                confluence_count += 1
            
            # Need at least 3 confluence factors for long
            if confluence_count >= 3 and rsi_1h_oversold:
                desired_signal = SIZE_STRONG if confluence_count >= 4 else SIZE_BASE
        
        # SHORT: Daily bear + 1h RSI overbought + (15m RSI overbought OR BB touch)
        elif daily_bear:
            confluence_count = 0
            if rsi_1h_overbought:
                confluence_count += 1
            if rsi_15m_overbought:
                confluence_count += 1
            if touch_upper:
                confluence_count += 1
            if not above_sma50:
                confluence_count += 1
            
            # Need at least 3 confluence factors for short
            if confluence_count >= 3 and rsi_1h_overbought:
                desired_signal = -SIZE_STRONG if confluence_count >= 4 else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_15m[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals