#!/usr/bin/env python3
"""
Experiment #681: 15m Primary + 1h/4h/1d HTF — Scoring-Based Multi-Confluence System

Hypothesis: Previous 15m strategies failed with Sharpe=0.000 (ZERO TRADES) because entry
conditions were TOO STRICT (requiring ALL filters to agree). This strategy uses a SCORING
system where partial confluence can trigger entries, ensuring trade generation while
maintaining selectivity.

Key innovations:
1. SCORING SYSTEM: Each condition adds points, entry at threshold (not all required)
2. 1d HMA(21): Regime bias (+2 pts if aligned)
3. 4h HMA(21): Intermediate trend confirmation (+2 pts if aligned)
4. 15m RSI(7): Fast mean-reversion entry trigger (+2 pts at extremes)
5. Volume spike filter: Only trade when volume > 1.5x 20-bar avg (+1 pt)
6. Session bias: London/NY overlap (00-12 UTC) gets +1 pt
7. ATR(14) trailing stop: 2.5x for risk management
8. Size: 0.15-0.25 (smaller for 15m frequency, target 50-80 trades/year)

Why this should work on 15m:
- RSI(7) extremes occur ~10-15% of bars, enough for entries
- Scoring allows entry with 3/5 conditions (not 5/5)
- HTF filters prevent counter-trend trades in strong regimes
- Volume filter avoids low-liquidity whipsaws

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-50%
Timeframe: 15m
Size: 0.15-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_score_rsi_htf_volume_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Pad first element
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

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_volume_avg(volume, period=20):
    """Rolling average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    dt = pd.to_datetime(open_time, unit='ms', utc=True)
    return dt.hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, period=7)  # Fast RSI for entries
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SCORING SYSTEM ===
        long_score = 0
        short_score = 0
        
        # 1. HTF REGIME BIAS (1d HMA) - +2 pts
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        if htf_bull:
            long_score += 2
        if htf_bear:
            short_score += 2
        
        # 2. INTERMEDIATE TREND (4h HMA) - +2 pts
        int_bull = close[i] > hma_4h_aligned[i]
        int_bear = close[i] < hma_4h_aligned[i]
        if int_bull:
            long_score += 2
        if int_bear:
            short_score += 2
        
        # 3. RSI ENTRY TRIGGER (RSI 7) - +2 pts at extremes
        rsi_oversold = rsi[i] < 30  # Oversold for long
        rsi_overbought = rsi[i] > 70  # Overbought for short
        if rsi_oversold:
            long_score += 2
        if rsi_overbought:
            short_score += 2
        
        # 4. VOLUME SPIKE - +1 pt
        vol_spike = False
        if not np.isnan(vol_avg[i]) and vol_avg[i] > 1e-10:
            vol_spike = volume[i] > 1.5 * vol_avg[i]
        if vol_spike:
            long_score += 1
            short_score += 1
        
        # 5. SESSION BIAS (00-12 UTC = London/NY overlap) - +1 pt
        utc_hour = get_utc_hour(open_time[i])
        session_active = 0 <= utc_hour < 12
        if session_active:
            long_score += 1
            short_score += 1
        
        # === ENTRY LOGIC (SCORING THRESHOLD) ===
        # Need score >= 5 for entry (allows 3/5 conditions without volume/session)
        desired_signal = 0.0
        
        if long_score >= 5 and htf_bull:
            # Strong long: HTF bias required
            if long_score >= 7:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        if short_score >= 5 and htf_bear:
            # Strong short: HTF bias required
            if short_score >= 7:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
                entry_atr = atr[i]
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