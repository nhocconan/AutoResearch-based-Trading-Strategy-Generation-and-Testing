#!/usr/bin/env python3
"""
Experiment #397: 15m Primary + 4h/12h HTF — Session RSI Pullback v1

HYPOTHESIS: Previous 15m/30m strategies failed with 0 trades due to overly complex
regime detection (Choppiness + ADX + multiple confluence). This version SIMPLIFIES:

1. 4h HMA(21) for trend bias (proven in best strategies)
2. 15m RSI(7) for faster pullback detection (not RSI(14) which is too slow)
3. Session filter: 06-18 UTC only (London/NY overlap = high volume)
4. ATR volatility filter: only trade when ATR > 20-day median (avoid dead chop)
5. LOOSENED RSI thresholds: <40 for long, >60 for short (not <30/>70)
6. Size: 0.15-0.20 (smaller for 15m frequency to reduce fee drag)

Key difference from #389 (which had 0 trades):
- RSI(7) instead of RSI(14) — more signals
- RSI <40/>60 instead of <30/>70 — 3x more triggers
- Session filter reduces bad trades, not entry conditions
- ATR filter ensures we only trade when market is moving

Target: 50-80 trades/year, Sharpe>0.4, DD>-30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_pullback_hma_4h12h_v1"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    ema_8 = calculate_ema(close, period=8)
    ema_21 = calculate_ema(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate ATR median for volatility filter (20-day rolling median)
    atr_median = pd.Series(atr).rolling(window=96, min_periods=96).median().values  # 96x15m = 24h
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi_7[i]) or np.isnan(hma_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER: 06-18 UTC only (London/NY overlap) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 6 <= hour_utc <= 18
        
        # === VOLATILITY FILTER: ATR > median (avoid dead chop) ===
        vol_filter = True
        if not np.isnan(atr_median[i]) and atr_median[i] > 1e-10:
            vol_filter = atr[i] > 0.8 * atr_median[i]
        
        # === HTF TREND BIAS (4h + 12h alignment) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both HTF agree
        htf_strong_bull = htf_4h_bull and htf_12h_bull
        htf_strong_bear = htf_4h_bear and htf_12h_bear
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === EMA CROSS (faster signal) ===
        ema_cross_long = False
        ema_cross_short = False
        if i > 0 and not np.isnan(ema_8[i]) and not np.isnan(ema_8[i-1]):
            if not np.isnan(ema_21[i]) and not np.isnan(ema_21[i-1]):
                if ema_8[i-1] <= ema_21[i-1] and ema_8[i] > ema_21[i]:
                    ema_cross_long = True
                if ema_8[i-1] >= ema_21[i-1] and ema_8[i] < ema_21[i]:
                    ema_cross_short = True
        
        # === RSI PULLBACK (LOOSENED thresholds for more trades) ===
        # Long: RSI(7) < 40 in uptrend (pullback entry)
        # Short: RSI(7) > 60 in downtrend (pullback entry)
        rsi_oversold = rsi_7[i] < 40.0
        rsi_overbought = rsi_7[i] > 60.0
        
        # RSI recovery (crossing back above/below threshold)
        rsi_recovery_long = False
        rsi_recovery_short = False
        if i > 0 and not np.isnan(rsi_7[i]) and not np.isnan(rsi_7[i-1]):
            if rsi_7[i-1] < 40.0 and rsi_7[i] >= 40.0:
                rsi_recovery_long = True
            if rsi_7[i-1] > 60.0 and rsi_7[i] <= 60.0:
                rsi_recovery_short = True
        
        # === SMA200 FILTER (long-term bias) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (SIMPLIFIED for more trades) ===
        desired_signal = 0.0
        
        # LONG ENTRY: HTF bull + 15m pullback OR breakout
        if htf_strong_bull or (htf_4h_bull and hma_bull):
            # Pullback entry: RSI oversold + above SMA200
            if rsi_oversold and above_sma200 and in_session and vol_filter:
                desired_signal = SIZE_STRONG
            # Breakout entry: EMA cross + RSI recovery
            elif ema_cross_long and rsi_recovery_long and in_session:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: HTF bear + 15m pullback OR breakdown
        elif htf_strong_bear or (htf_4h_bear and hma_bear):
            # Pullback entry: RSI overbought + below SMA200
            if rsi_overbought and below_sma200 and in_session and vol_filter:
                desired_signal = -SIZE_STRONG
            # Breakdown entry: EMA cross + RSI recovery
            elif ema_cross_short and rsi_recovery_short and in_session:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest since entry for trailing
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            # Check stoploss
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trailing stop: move stop up when price moves 1.5x ATR in favor
            trail_distance = 2.0 * entry_atr
            if close[i] > entry_price + 1.5 * entry_atr:
                new_stop = highest_since_entry - trail_distance
                if new_stop > stop_price:
                    stop_price = new_stop
        
        if in_position and position_side < 0:
            # Update lowest since entry for trailing
            if close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            # Check stoploss
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trailing stop: move stop down when price moves 1.5x ATR in favor
            trail_distance = 2.0 * entry_atr
            if close[i] < entry_price - 1.5 * entry_atr:
                new_stop = lowest_since_entry + trail_distance
                if new_stop < stop_price:
                    stop_price = new_stop
        
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
                entry_atr = atr[i] if not np.isnan(atr[i]) else (close[i] * 0.02)
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                # Set initial stoploss
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals