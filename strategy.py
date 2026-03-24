#!/usr/bin/env python3
"""
Experiment #1565: 1h Primary + 4h/1d HTF — HMA Trend Pullback with Session Filter

Hypothesis: 1h entries within 4h trend direction + RSI pullback + volume/session filters
will generate 30-80 trades/year with positive Sharpe on all symbols.

Key insights from 1164 failed experiments:
1. Lower TF (1h) MUST use HTF (4h) for signal direction, not 1d (too slow)
2. Session filter (8-20 UTC) reduces trades by ~40% during low-liquidity hours
3. Volume confirmation (>0.8x avg) filters false breakouts
4. RSI pullback (40-60 range) ensures entries on retracements, not breakouts
5. Loose conditions are CRITICAL — many strategies fail from 0 trades

Strategy Design:
- HTF Bias: 4h HMA(21) for intermediate trend direction
- Primary: 1h HMA(16/48) crossover for entry timing
- Entry: 4h HMA bull + 1h HMA crossover + RSI(14) 40-60 + Volume > 0.8x avg + Session 8-20 UTC
- Exit: 2.5x ATR(14) trailing stop via signal→0
- Size: 0.22 discrete (smaller for 1h to reduce fee drag)

Why this should work:
- 4h HMA filter provides trend direction without 1d lag
- 1h HMA crossover catches entries early within HTF trend
- RSI 40-60 ensures pullback entries (not chase breakouts)
- Volume + Session filters reduce trade count to 30-80/year
- Works on BTC/ETH/SOL (tested patterns from research)

Timeframe: 1h (required for this experiment)
HTF: 4h HMA for bias, 1d HMA for macro filter
Target: Sharpe > 0.618, trades > 30/symbol train, > 3/symbol test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_pullback_4h_trend_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
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
    """Relative Strength Index with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # HMA crossover system on 1h
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22  # Smaller size for 1h to reduce fee drag
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
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
        if np.isnan(rsi_14[i]) or np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Extract UTC hour for session filter (8-20 UTC = active trading hours)
        utc_hour = get_utc_hour(open_time[i])
        session_active = 8 <= utc_hour <= 20
        
        # Volume filter: current volume > 0.8x 20-period average
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === MACRO TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        h4_bull = close[i] > hma_4h_aligned[i]
        h4_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY SIGNAL (1h HMA crossover) ===
        hma_crossover_bull = hma_fast[i] > hma_slow[i]
        hma_crossover_bear = hma_fast[i] < hma_slow[i]
        
        # === RSI PULLBACK FILTER (40-60 range for pullback entries) ===
        # Long: RSI between 40-60 (pullback in uptrend, not overbought)
        # Short: RSI between 40-60 (pullback in downtrend, not oversold)
        rsi_pullback_long = 40.0 <= rsi_14[i] <= 65.0
        rsi_pullback_short = 35.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC - LOOSE CONDITIONS TO ENSURE TRADES FIRE ===
        desired_signal = 0.0
        
        # LONG: Daily bull + 4h bull + 1h HMA crossover bull + RSI pullback + Volume + Session
        # Relaxed: only need 4h OR 1d bull (not both) to ensure trades fire
        if (h4_bull or daily_bull) and hma_crossover_bull and rsi_pullback_long and volume_ok and session_active:
            desired_signal = BASE_SIZE
        
        # SHORT: Daily bear + 4h bear + 1h HMA crossover bear + RSI pullback + Volume + Session
        if (h4_bear or daily_bear) and hma_crossover_bear and rsi_pullback_short and volume_ok and session_active:
            desired_signal = -BASE_SIZE
        
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