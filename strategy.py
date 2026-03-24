#!/usr/bin/env python3
"""
Experiment #1630: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + CHOP Regime + Volume/Session

Hypothesis: After repeated 1h failures (exp #1620, #1625, #1628 all got 0 trades),
the key is using HTF for DIRECTION and 1h only for precise ENTRY TIMING.

Previous 1h failures caused by:
- Too many trades (>200/year) → fee drag kills profit
- Not enough HTF confluence → whipsaw entries
- CRSI too strict → 0 trades

This strategy uses:
1. 12h HMA(21) for PRIMARY trend bias (slower, more reliable than 4h)
2. 4h CHOP(14) for regime detection (trend vs range)
3. 1h RSI(14) for entry timing (pullback in HTF trend direction)
4. Volume filter (only trade when vol > 1.0x 20-bar avg)
5. Session filter (only 8-20 UTC for institutional flow)
6. ATR(14) trailing stop at 2.5x

Key insight: 1h should generate 30-60 trades/year, not 200+.
Use strict confluence: ALL filters must align before entry.

Timeframe: 1h (required for this experiment)
HTF: 12h HMA for bias, 4h CHOP for regime (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
Position Size: 0.20 (smaller for lower TF to reduce fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_4h12h_vol_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if loss_smooth[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i-1] / loss_smooth[i-1]))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h CHOP for regime
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1h HMA for entry timing
    hma_fast = calculate_hma(close, period=8)
    hma_slow = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # Smaller size for 1h to reduce fee drag
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = extract_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 1e-10 else 0.0
        vol_confirmed = vol_ratio > 0.8  # At least 80% of average volume
        
        # === REGIME DETECTION (4h CHOP) ===
        # CHOP > 55 = choppy/range, CHOP < 45 = trending
        is_choppy = chop_4h_aligned[i] > 55.0
        is_trending = chop_4h_aligned[i] < 45.0
        
        # === TREND BIAS (12h HMA) ===
        daily_bull = close[i] > hma_12h_aligned[i]
        daily_bear = close[i] < hma_12h_aligned[i]
        
        # === 1h HMA CROSSOVER ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # Check previous bar for crossover detection
        hma_bull_prev = False
        hma_bear_prev = False
        if i > 0 and not np.isnan(hma_fast[i-1]) and not np.isnan(hma_slow[i-1]):
            hma_bull_prev = hma_fast[i-1] > hma_slow[i-1]
            hma_bear_prev = hma_fast[i-1] < hma_slow[i-1]
        
        # Bullish crossover (fast crosses above slow)
        hma_cross_up = hma_bull and not hma_bull_prev
        # Bearish crossover (fast crosses below slow)
        hma_cross_down = hma_bear and not hma_bear_prev
        
        # === RSI ENTRY ZONES ===
        # For longs: RSI pullback to 30-45 zone (oversold but not extreme)
        rsi_long_zone = 30.0 < rsi[i] < 50.0
        # For shorts: RSI bounce to 50-70 zone (overbought but not extreme)
        rsi_short_zone = 50.0 < rsi[i] < 70.0
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # Only trade during session hours with volume confirmation
        if in_session and vol_confirmed:
            # REGIME 1: TRENDING MARKET - Trend Following
            if is_trending:
                # Long: 12h bullish + 1h HMA bullish + RSI in long zone
                if daily_bull and hma_bull and rsi_long_zone:
                    desired_signal = BASE_SIZE
                # Short: 12h bearish + 1h HMA bearish + RSI in short zone
                elif daily_bear and hma_bear and rsi_short_zone:
                    desired_signal = -BASE_SIZE
            
            # REGIME 2: CHOPPY MARKET - Mean Reversion with HMA crossover
            elif is_choppy:
                # Long on bullish crossover with RSI pullback + 12h neutral/bull
                if hma_cross_up and rsi_long_zone and not daily_bear:
                    desired_signal = BASE_SIZE
                # Short on bearish crossover with RSI bounce + 12h neutral/bear
                elif hma_cross_down and rsi_short_zone and not daily_bull:
                    desired_signal = -BASE_SIZE
                # Hold existing position if already in trade
                elif in_position:
                    desired_signal = BASE_SIZE if position_side > 0 else -BASE_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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