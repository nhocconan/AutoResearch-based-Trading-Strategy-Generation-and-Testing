#!/usr/bin/env python3
"""
Experiment #1030: 1h Primary + 4h/12h HTF — Simplified Trend Pullback with Session Filter

Hypothesis: After 747+ failed strategies, the pattern is clear:
1. Complex regime detection (Fisher, Choppiness) often leads to 0 trades or negative Sharpe
2. Simpler trend-following with pullback entries works better across all market conditions
3. 1h timeframe needs STRICT filters to avoid fee drag (>100 trades/year kills profit)

Strategy components:
1. 12h HMA21: Major trend direction (only long when price > 12h HMA, only short when <)
2. 4h HMA21: Medium-term confirmation (slope must align with 12h trend)
3. 1h RSI(14): Pullback entry (RSI 35-45 for longs in uptrend, 55-65 for shorts in downtrend)
4. Session filter: Only trade 8-20 UTC (reduces trades by ~50%, focuses on liquid hours)
5. Volume filter: Current volume > 0.8x 20-bar average (confirms participation)
6. ATR(14) trailing stop: 2.5x ATR for risk management

Why this works:
- HTF trend filter ensures we trade WITH the trend (not counter-trend)
- RSI pullback entries catch retracements within the trend (better R:R than breakouts)
- Session + volume filters reduce trade count to target 30-60/year on 1h
- Simple logic = fewer edge cases = more consistent across BTC/ETH/SOL

Critical fixes from failures:
- SIMPLER than Fisher/Choppiness (those got 0 trades or negative Sharpe)
- RELAXED RSI thresholds (35-45 not 30-35) to ensure trades generate
- Session filter prevents overnight whipsaw (major cause of drawdown)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year with filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_trend_pullback_4h12h_hma_rsi_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average."""
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope (positive = uptrend, negative = downtrend)."""
    n = len(hma_values)
    slope = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i-lookback]):
            slope[i] = (hma_values[i] - hma_values[i-lookback]) / (hma_values[i-lookback] + 1e-10)
    
    return slope

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array // 1000) // 3600) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA21 for major trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h HMA21 for medium-term confirmation
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 4h HMA slope for trend confirmation
    hma_4h_slope = calculate_hma_slope(hma_4h_aligned, lookback=3)
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Volume filter: 20-bar rolling average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: extract UTC hour
    utc_hours = get_hour_from_open_time(open_time)
    in_session = (utc_hours >= 8) & (utc_hours <= 20)
    
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
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(hma_4h_slope[i]) or np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        
        # === MACRO TREND (12h HMA21) ===
        major_bull = close[i] > hma_12h_aligned[i]
        major_bear = close[i] < hma_12h_aligned[i]
        
        # === MEDIUM TREND (4h HMA21 + slope) ===
        medium_bull = close[i] > hma_4h_aligned[i] and hma_4h_slope[i] > 0.001
        medium_bear = close[i] < hma_4h_aligned[i] and hma_4h_slope[i] < -0.001
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI 35-50 in uptrend (pullback, not oversold crash)
        # Short: RSI 50-65 in downtrend (pullback, not overbought spike)
        rsi_long_pullback = 35 <= rsi_1h[i] <= 50
        rsi_short_pullback = 50 <= rsi_1h[i] <= 65
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Must have: major bullish + medium bullish + volume + session + RSI pullback
        if major_bull and medium_bull and volume_ok and session_ok:
            if rsi_long_pullback:
                desired_signal = BASE_SIZE
            elif rsi_1h[i] < 35 and volume_ok:
                # Deep pullback entry (higher risk, reduced size)
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Must have: major bearish + medium bearish + volume + session + RSI pullback
        if major_bear and medium_bear and volume_ok and session_ok:
            if rsi_short_pullback:
                desired_signal = -BASE_SIZE
            elif rsi_1h[i] > 65 and volume_ok:
                # Deep pullback entry (higher risk, reduced size)
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if major trend still bullish
                if major_bull and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if major trend still bearish
                if major_bear and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if major trend reverses OR RSI extremely overbought
            if not major_bull or rsi_1h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if major trend reverses OR RSI extremely oversold
            if not major_bear or rsi_1h[i] < 25:
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