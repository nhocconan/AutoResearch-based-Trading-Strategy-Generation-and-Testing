#!/usr/bin/env python3
"""
Experiment #393: 5m Primary + 15m/4h HTF — Session-Filtered Momentum Breakout

Hypothesis: 5m timeframe is unexplored (0 experiments). Key insight: 5m generates
too many false signals without strict filters. This strategy uses:
1. 4h HMA for primary trend bias (only trade in HTF direction)
2. 15m RSI for momentum confirmation (avoid entering against momentum)
3. 5m for precise entry timing (breakout + volume spike)
4. Session filter: 08:00-20:00 UTC only (London/NY overlap = high liquidity)
5. Small position size (0.15-0.20) due to higher trade frequency

Why this might work on 5m:
- HTF trend filter prevents counter-trend trades (major failure mode)
- Session filter avoids Asian session whipsaws (low liquidity)
- Volume confirmation reduces false breakouts
- Small size + strict entries = manageable fee drag

Target: 50-120 trades/year, Sharpe>0.4, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_momentum_breakout_15m4h_v1"
timeframe = "5m"
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs SMA"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_sma
    vol_ratio[vol_sma < 1e-10] = np.nan
    
    return vol_ratio

def calculate_momentum(close, period=10):
    """Rate of Change momentum"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    mom = np.zeros(n)
    mom[:] = np.nan
    for i in range(period, n):
        if close[i-period] > 1e-10:
            mom[i] = (close[i] - close[i-period]) / close[i-period] * 100.0
    
    return mom

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    import datetime
    ts_seconds = open_time / 1000.0
    dt = datetime.datetime.utcfromtimestamp(ts_seconds)
    return dt.hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    # Calculate primary (5m) indicators
    hma_5m = calculate_hma(close, period=21)
    hma_5m_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi_5m = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)
    momentum = calculate_momentum(close, 10)
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_5m[i]) or np.isnan(rsi_5m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08:00-20:00 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === 4h TREND BIAS (HTF) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m MOMENTUM CONFIRMATION ===
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        rsi_15m_neutral = 40.0 <= rsi_15m_aligned[i] <= 60.0
        rsi_15m_bull = rsi_15m_aligned[i] > 50.0
        rsi_15m_bear = rsi_15m_aligned[i] < 50.0
        
        # === 5m HMA TREND ===
        hma_5m_bull = close[i] > hma_5m[i]
        hma_5m_bear = close[i] < hma_5m[i]
        
        # === HMA CROSSOVER (5m) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_5m_fast[i]) and not np.isnan(hma_5m_fast[i-1]):
            if not np.isnan(hma_5m[i]) and not np.isnan(hma_5m[i-1]):
                if hma_5m_fast[i-1] <= hma_5m[i-1] and hma_5m_fast[i] > hma_5m[i]:
                    hma_cross_long = True
                if hma_5m_fast[i-1] >= hma_5m[i-1] and hma_5m_fast[i] < hma_5m[i]:
                    hma_cross_short = True
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = False
        if not np.isnan(vol_ratio[i]):
            vol_spike = vol_ratio[i] > 1.5
        
        # === MOMENTUM CONFIRMATION ===
        mom_positive = False
        mom_negative = False
        if not np.isnan(momentum[i]):
            mom_positive = momentum[i] > 0.5
            mom_negative = momentum[i] < -0.5
        
        # === RSI EXTREMES (5m) ===
        rsi_5m_oversold = rsi_5m[i] < 35.0
        rsi_5m_overbought = rsi_5m[i] > 65.0
        rsi_5m_neutral = 40.0 <= rsi_5m[i] <= 60.0
        
        # === ENTRY LOGIC (STRICT - multiple confluence required) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m bull + 5m breakout + volume + session
        if in_session and htf_4h_bull and htf_15m_bull:
            # Entry: HMA cross OR (breakout above SMA50 + volume + momentum)
            if hma_cross_long:
                if rsi_5m_neutral or rsi_5m[i] > 45.0:
                    desired_signal = SIZE_STRONG if vol_spike else SIZE_BASE
            elif above_sma50 and vol_spike and mom_positive:
                if rsi_5m_neutral:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 15m bear + 5m breakdown + volume + session
        elif in_session and htf_4h_bear and htf_15m_bear:
            # Entry: HMA cross OR (breakdown below SMA50 + volume + momentum)
            if hma_cross_short:
                if rsi_5m_neutral or rsi_5m[i] < 55.0:
                    desired_signal = -SIZE_STRONG if vol_spike else -SIZE_BASE
            elif below_sma50 and vol_spike and mom_negative:
                if rsi_5m_neutral:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update trailing high
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            # Check stoploss
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Update trailing low
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            # Check stoploss
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
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals