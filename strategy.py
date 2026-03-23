#!/usr/bin/env python3
"""
Experiment #1150: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume/Session Filter

Hypothesis: After 838+ failed experiments, the pattern is clear:
- CRSI + Choppiness combinations FAIL (negative Sharpe in #1137-#1148)
- Complex regime switching causes 0 trades (#1148 Sharpe=0.000)
- SIMPLE trend + pullback works (HMA+RSI baseline had positive returns)
- Lower TF (1h) needs VERY STRICT filters to avoid fee drag (>100 trades/yr = fail)

This strategy uses PROVEN components with STRICT confluence for 1h:
1. 12h HMA(21) for macro trend direction (slow, few false signals)
2. 4h HMA(21) for intermediate trend confirmation
3. 1h RSI(14) pullback entry (RSI 35-45 for long, 55-65 for short)
4. Volume filter: current volume > 1.2x 20-bar average (confirms moves)
5. Session filter: only 8-20 UTC (high liquidity hours)
6. ATR(14) 2.5x trailing stop (wider for lower TF noise)
7. Position size 0.25 discrete (conservative for 1h frequency)

Why this should beat Sharpe=0.612:
- 12h HMA prevents counter-trend trades that destroyed 2022 returns
- 4h HMA adds intermediate confirmation (3-filter confluence)
- RSI pullback (not breakout) = better entry prices, fewer whipsaws
- Volume + session filters cut false signals by ~60%
- Target: 40-70 trades/year on 1h (optimal for fee drag)

Timeframe: 1h (primary)
HTF: 4h, 12h — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base (discrete: 0.0, ±0.25)
Stoploss: 2.5x ATR trailing
Target: 40-70 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_volume_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Helper function for WMA
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.convolve(series, weights, mode='full')[:len(series)]
        result[:span-1] = np.nan
        return result
    
    close_series = pd.Series(close)
    
    # WMA(n/2)
    wma_half = wma(close, period // 2)
    # WMA(n)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2.0 * wma_half - wma_full
    
    # WMA of diff with sqrt(n)
    sqrt_period = int(np.sqrt(period))
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    n = len(volume)
    vol_avg = np.full(n, np.nan)
    
    if n < period:
        return vol_avg
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    # Convert to seconds, then to datetime
    timestamps = open_time / 1000.0
    # Use pandas to extract hour (UTC)
    hours = pd.to_datetime(timestamps, unit='s').hour.values
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
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h HMA for intermediate trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    hours = get_hour_from_open_time(open_time)
    
    # Also calculate 1h HMA for local trend
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(hma_1h[i]) or np.isnan(vol_avg[i]):
            continue
        if atr[i] <= 1e-10 or vol_avg[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # High liquidity hours, avoid Asian session noise
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        # Current volume > 1.2x 20-bar average confirms move
        volume_confirmed = volume[i] > 1.2 * vol_avg[i]
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        intermediate_bull = close[i] > hma_4h_aligned[i]
        intermediate_bear = close[i] < hma_4h_aligned[i]
        
        # === LOCAL TREND (1h HMA) ===
        local_bull = close[i] > hma_1h[i]
        local_bear = close[i] < hma_1h[i]
        
        # === RSI PULLBACK ENTRY (not breakout) ===
        # Long: RSI 35-45 (pullback in uptrend)
        # Short: RSI 55-65 (pullback in downtrend)
        rsi_pullback_long = 35.0 <= rsi_1h[i] <= 45.0
        rsi_pullback_short = 55.0 <= rsi_1h[i] <= 65.0
        
        # === EXTREME RSI EXIT ===
        # Exit long if RSI > 70 (overbought)
        # Exit short if RSI < 30 (oversold)
        rsi_extreme_long = rsi_1h[i] > 70.0
        rsi_extreme_short = rsi_1h[i] < 30.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + intermediate bull + RSI pullback + volume + session
        if macro_bull and intermediate_bull and rsi_pullback_long and volume_confirmed and in_session:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + intermediate bear + RSI pullback + volume + session
        elif macro_bear and intermediate_bear and rsi_pullback_short and volume_confirmed and in_session:
            desired_signal = -BASE_SIZE
        
        # === EXTREME RSI EXIT ===
        if in_position and position_side > 0 and rsi_extreme_long:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_extreme_short:
            desired_signal = 0.0
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
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
                # Hold long if macro and intermediate still bull
                if macro_bull and intermediate_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro and intermediate still bear
                if macro_bear and intermediate_bear:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals