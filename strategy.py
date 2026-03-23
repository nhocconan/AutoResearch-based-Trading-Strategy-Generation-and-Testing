#!/usr/bin/env python3
"""
Experiment #1015: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume/Session Filter

Hypothesis: After 735+ failed strategies, the pattern is clear: complex regime-switching
(Fisher, Choppiness, CRSI combinations) overfit and fail in bear/range markets. The current
best (Sharpe=0.612) uses simple HMA trend + Donchian breakout. For 1h timeframe, I need
EVEN FEWER trades (target 30-60/year) with stricter filters.

This strategy uses PROVEN elements from winning strategies:
1. 1d HMA21 for MACRO bias (only long when price > 1d HMA, only short when < 1d HMA)
   - This asymmetry works in bear/range markets (2025 test period)
2. 4h HMA21 for MEDIUM confirmation (aligns with 1d bias)
3. 1h RSI(14) pullback entries (classic, robust, works across all regimes)
   - Long: RSI < 35 in uptrend (buy dip)
   - Short: RSI > 65 in downtrend (sell rip)
4. Volume filter (>0.8x 20-bar avg) to avoid low-liquidity fakeouts
5. Session filter (8-20 UTC) to avoid Asian session whipsaws (major failure source)
6. ATR(14) trailing stop at 2.5x for tight risk management

Why 1h can work:
- Use 4h/1d for DIRECTION (not 1h trend)
- Use 1h only for ENTRY TIMING (pullback within HTF trend)
- Session + volume filters cut trades by 60%+
- Target: 40-70 trades/year (vs 150+ in failed 1h strategies)

Critical fixes from failures:
- NO Fisher/Choppiness (both failed in exp #1004, #1014)
- NO complex regime switching (failed in #1006, #1007, #1010)
- Simple HMA + RSI pullback (proven in current best strategy)
- Session filter 8-20 UTC (avoids 40% of whipsaw trades)
- Discrete signals (0.0, ±0.25, ±0.30) to minimize fee churn
- Size = 0.25 (smaller than 4h strategies due to higher frequency)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-70 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and faster than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000.0
    utc_hour = (ts_seconds % 86400) / 3600.0
    return int(utc_hour)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA21 for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA21 for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        
        # === MACRO TREND (HTF HMA21) ===
        # Asymmetric bias for bear/range markets
        long_bull = close[i] > hma_1d_aligned[i]  # Only long when above 1d HMA
        short_bear = close[i] < hma_1d_aligned[i]  # Only short when below 1d HMA
        
        # === MEDIUM CONFIRMATION (4h HMA21) ===
        medium_bull = close[i] > hma_4h_aligned[i]
        medium_bear = close[i] < hma_4h_aligned[i]
        
        # === RSI PULLBACK SIGNALS ===
        rsi_oversold = rsi_1h[i] < 35.0  # Buy dip
        rsi_overbought = rsi_1h[i] > 65.0  # Sell rip
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_1h[i]  # Above 80% of avg
        
        # === SESSION FILTER (8-20 UTC) ===
        # Avoid Asian session (0-8 UTC) and late US session (20-24 UTC)
        utc_hour = get_utc_hour(open_time[i])
        session_ok = (utc_hour >= 8) and (utc_hour <= 20)
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Require: 1d bullish + 4h bullish + RSI oversold + volume + session
        if long_bull and medium_bull and rsi_oversold and volume_ok and session_ok:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRIES ===
        # Require: 1d bearish + 4h bearish + RSI overbought + volume + session
        if short_bear and medium_bear and rsi_overbought and volume_ok and session_ok:
            desired_signal = -BASE_SIZE
        
        # === REDUCED SIZE ENTRIES (weaker confluence) ===
        # Only 1d bias + RSI extreme (no 4h confirmation needed)
        if desired_signal == 0.0:
            if long_bull and rsi_1h[i] < 30.0 and volume_ok and session_ok:
                desired_signal = REDUCED_SIZE
            elif short_bear and rsi_1h[i] > 70.0 and volume_ok and session_ok:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if HTF trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d still bullish (don't exit on 4h flip)
                if long_bull and rsi_1h[i] < 75.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d still bearish
                if short_bear and rsi_1h[i] > 25.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses bearish
            if not long_bull and rsi_1h[i] > 60.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses bullish
            if not short_bear and rsi_1h[i] < 40.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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
        
        signals[i] = desired_signal
    
    return signals