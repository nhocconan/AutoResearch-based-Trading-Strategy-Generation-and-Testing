#!/usr/bin/env python3
"""
Experiment #501: 15m Primary + 1h/4h HTF — Simple Trend Pullback Strategy

Hypothesis: 15m timeframe has ZERO successful experiments because strategies are TOO COMPLEX.
This uses SIMPLE logic: 4h HMA trend + 1h RSI momentum + 15m RSI pullback entry.
Key insight from failures: complex regime filters (CHOP, CRSI) generate 0 trades on 15m.

Strategy logic:
1. 4h HMA(21) = primary trend bias (HTF filter, slow and reliable)
2. 1h RSI(14) = momentum confirmation (medium speed, avoids counter-trend)
3. 15m RSI(7) = entry timing on pullbacks (fast, catches dips in uptrend)
4. Session filter: 00-12 UTC only (London+NY overlap, highest liquidity)
5. ATR(14)*2.5 stoploss on all positions
6. Discrete signal sizes: 0.0, ±0.20, ±0.30

Why this should work on 15m:
- LOOSE entry thresholds (RSI<35/65 not 30/70) = MORE trades
- Only 3 filters (HTF trend + 1h RSI + 15m RSI) = not overfiltered
- Session filter reduces noise, not trades (crypto active 24/7 anyway)
- Target: 50-100 trades/year = 200-400 over 4 year train

CRITICAL: Call get_htf_data() ONCE before loop, use aligned arrays inside.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_4h1h_v1"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1h RSI for momentum
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    ema_15m = calculate_ema(close, period=50)
    atr = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=7)  # Faster RSI for 15m entries
    rsi_15m_std = calculate_rsi(close, period=14)  # Standard RSI for confirmation
    
    # Extract session hours
    session_hours = np.array([get_session_hour(ot) for ot in open_time])
    
    # Session filter: 00-12 UTC (London + NY overlap)
    in_session = (session_hours >= 0) & (session_hours <= 12)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(rsi_15m_std[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4h HTF TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1h MOMENTUM CONFIRMATION ===
        mom_bull = rsi_1h_aligned[i] > 50.0
        mom_bear = rsi_1h_aligned[i] < 50.0
        mom_strong_bull = rsi_1h_aligned[i] > 55.0
        mom_strong_bear = rsi_1h_aligned[i] < 45.0
        
        # === 15m RSI PULLBACK ENTRY (LOOSE thresholds for more trades) ===
        rsi_oversold = rsi_15m[i] < 40.0
        rsi_overbought = rsi_15m[i] > 60.0
        rsi_extreme_oversold = rsi_15m[i] < 35.0
        rsi_extreme_overbought = rsi_15m[i] > 65.0
        
        # RSI turning up/down
        rsi_rising = (i > 0 and not np.isnan(rsi_15m[i-1]) and rsi_15m[i] > rsi_15m[i-1])
        rsi_falling = (i > 0 and not np.isnan(rsi_15m[i-1]) and rsi_15m[i] < rsi_15m[i-1])
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === EMA FILTER ===
        above_ema50 = close[i] > ema_15m[i] if not np.isnan(ema_15m[i]) else False
        below_ema50 = close[i] < ema_15m[i] if not np.isnan(ema_15m[i]) else False
        
        # === VOLATILITY FILTER ===
        if i >= 100:
            atr_mean = np.nanmean(atr[max(0,i-100):i])
            atr_ratio = atr[i] / atr_mean if atr_mean > 0 else 1.0
        else:
            atr_ratio = 1.0
        vol_normal = atr_ratio < 3.0  # Allow higher vol on 15m
        
        # === SESSION FILTER ===
        # Only trade during high-liquidity hours (but allow exits anytime)
        active_session = in_session[i]
        
        # === ENTRY LOGIC (LOOSE - generate trades!) ===
        desired_signal = 0.0
        
        # TREND LONG: 4h bull + 1h mom + 15m RSI pullback
        if htf_bull and mom_bull and vol_normal:
            if rsi_extreme_oversold and rsi_rising:
                # Deep oversold + turning up = strong long
                desired_signal = SIZE_STRONG if active_session else SIZE_BASE
            elif rsi_oversold and rsi_rising and above_ema50:
                # Moderate oversold + above EMA50 = standard long
                desired_signal = SIZE_BASE if active_session else SIZE_BASE * 0.5
            elif rsi_15m[i] < 45.0 and rsi_15m_std[i] < 50.0 and rsi_rising:
                # RSI(7) < 45 + RSI(14) < 50 + rising = pullback entry
                desired_signal = SIZE_BASE if active_session else SIZE_BASE * 0.5
        
        # TREND SHORT: 4h bear + 1h mom + 15m RSI bounce
        elif htf_bear and mom_bear and vol_normal:
            if rsi_extreme_overbought and rsi_falling:
                # Deep overbought + turning down = strong short
                desired_signal = -SIZE_STRONG if active_session else -SIZE_BASE
            elif rsi_overbought and rsi_falling and below_ema50:
                # Moderate overbought + below EMA50 = standard short
                desired_signal = -SIZE_BASE if active_session else -SIZE_BASE * 0.5
            elif rsi_15m[i] > 55.0 and rsi_15m_std[i] > 50.0 and rsi_falling:
                # RSI(7) > 55 + RSI(14) > 50 + falling = bounce entry
                desired_signal = -SIZE_BASE if active_session else -SIZE_BASE * 0.5
        
        # MEAN REVERSION (works in any HTF regime, but smaller size)
        if desired_signal == 0.0 and vol_normal:
            if rsi_extreme_oversold and rsi_rising and above_ema50:
                # Extreme oversold + rising + above EMA = MR long
                desired_signal = SIZE_BASE * 0.8 if active_session else SIZE_BASE * 0.4
            elif rsi_extreme_overbought and rsi_falling and below_ema50:
                # Extreme overbought + falling + below EMA = MR short
                desired_signal = -SIZE_BASE * 0.8 if active_session else -SIZE_BASE * 0.4
        
        # HMA CROSSOVER (additional entry signal)
        if desired_signal == 0.0 and vol_normal and i > 0:
            hma_cross_bull = (close[i] > hma_15m[i]) and (close[i-1] <= hma_15m[i-1])
            hma_cross_bear = (close[i] < hma_15m[i]) and (close[i-1] >= hma_15m[i-1])
            
            if hma_cross_bull and htf_bull and mom_bull:
                desired_signal = SIZE_BASE * 0.8 if active_session else SIZE_BASE * 0.4
            elif hma_cross_bear and htf_bear and mom_bear:
                desired_signal = -SIZE_BASE * 0.8 if active_session else -SIZE_BASE * 0.4
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest since entry for trailing
            highest_since_entry = max(highest_since_entry, high[i])
            # Check stoploss
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trail stop: move up as price rises
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            # Update lowest since entry for trailing
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Check stoploss
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trail stop: move down as price falls
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        elif desired_signal > 0:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal < 0:
            final_signal = -SIZE_BASE * 0.5
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
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals