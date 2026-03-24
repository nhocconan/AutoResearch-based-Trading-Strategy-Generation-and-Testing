#!/usr/bin/env python3
"""
Experiment #410: 1h Primary + 4h/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Recent failures (#398-#409) had 0 trades due to overly complex regime
detection and too-strict entry conditions. This version SIMPLIFIES while keeping
proven MTF structure (4h HMA trend + 1h RSI pullback entries).

Key changes from failed experiments:
1. SIMPLER regime: Just 4h HMA slope + Choppiness (not ADX+CHOP+multiple filters)
2. RSI thresholds: 35/65 (not 25/75 which is too extreme for 1h)
3. Session filter: 08-20 UTC for liquidity (proven in baseline)
4. Fewer confluence requirements: max 3 filters per entry
5. Ensure at least ONE entry path is always available (HMA cross OR RSI pullback)

Entry Logic:
- Long: 4h HMA bull + (1h RSI<40 OR 1h HMA cross up) + Session + CHOP<65
- Short: 4h HMA bear + (1h RSI>60 OR 1h HMA cross down) + Session + CHOP<65

Position sizing: 0.20 base, 0.30 when 1d HTF aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.40, DD>-35%, trades>=40 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_session_v2"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    hma_1h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
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
        
        # === SESSION FILTER (08-20 UTC for liquidity) ===
        # Extract hour from open_time (assuming milliseconds timestamp)
        open_time = prices["open_time"].iloc[i]
        hour_utc = (open_time // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF CONFIRMATION (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 1h HMA TREND ===
        hma_bull = close[i] > hma_1h[i]
        hma_bear = close[i] < hma_1h[i]
        
        # === HMA CROSSOVER (1h) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_1h_fast[i]) and not np.isnan(hma_1h_fast[i-1]):
            if not np.isnan(hma_1h[i]) and not np.isnan(hma_1h[i-1]):
                if hma_1h_fast[i-1] <= hma_1h[i-1] and hma_1h_fast[i] > hma_1h[i]:
                    hma_cross_long = True
                if hma_1h_fast[i-1] >= hma_1h[i-1] and hma_1h_fast[i] < hma_1h[i]:
                    hma_cross_short = True
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CHOPPINESS FILTER (avoid choppy markets for trend entries) ===
        not_choppy = chop[i] < 65.0
        
        # === RSI PULLBACK (LOOSENED for more trades) ===
        rsi_pullback_long = rsi[i] < 40.0
        rsi_pullback_short = rsi[i] > 60.0
        
        # === ENTRY LOGIC (SIMPLIFIED - ensure trades generate) ===
        desired_signal = 0.0
        
        # LONG entries (3 confluence: HTF bull + entry trigger + session/not choppy)
        if htf_4h_bull and in_session:
            # Path 1: RSI pullback in uptrend
            if rsi_pullback_long and above_sma200:
                desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
            # Path 2: HMA crossover (always available)
            elif hma_cross_long and not_choppy:
                desired_signal = SIZE_BASE
        
        # SHORT entries (3 confluence: HTF bear + entry trigger + session/not choppy)
        elif htf_4h_bear and in_session:
            # Path 1: RSI pullback in downtrend
            if rsi_pullback_short and below_sma200:
                desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
            # Path 2: HMA crossover (always available)
            elif hma_cross_short and not_choppy:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
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
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals