#!/usr/bin/env python3
"""
Experiment #333: 5m Primary + 15m/4h HTF — Session-Filtered Momentum Pullback

Hypothesis: 5m timeframe is unexplored territory. Key insight: 5m is TOO NOISY for
standalone signals. Must use 4h for STRONG trend bias, 15m for confirmation,
5m only for precise entry timing. Session filter CRITICAL (avoid Asian low-liquidity).

Strategy Design:
- 4h HMA(50): Primary trend bias (ONLY trade in HTF direction)
- 15m HMA(21): Intermediate trend confirmation
- 5m RSI(7): Fast RSI for entry timing (pullback in trend)
- 5m Volume: Confirm with volume spike (>1.2x 20-bar avg)
- Session Filter: 06-22 UTC only (London + NY sessions, skip Asian chop)
- Stoploss: 2.0x ATR(14) from entry

Why this might work:
1. HTF alignment prevents counter-trend trades (major failure mode on 5m)
2. Session filter avoids low-liquidity whipsaws
3. Fast RSI(7) catches pullbacks quickly without waiting for extreme levels
4. Volume confirmation filters false breakouts
5. Small size (0.15) accounts for higher fee drag on 5m

Target: 60-100 trades/year, Sharpe>0.40, DD>-30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_hma_rsi_vol_15m4h_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMA for trend bias
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (5m) indicators
    hma_5m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Fast RSI for 5m
    rsi_std = calculate_rsi(close, period=14)  # Standard RSI for confirmation
    vol_sma = calculate_sma(volume, 20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE = 0.15  # Small size for 5m (fee drag)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_5m[i]) or np.isnan(rsi[i]) or np.isnan(rsi_std[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (06-22 UTC) ===
        # Extract hour from open_time (assuming milliseconds timestamp)
        open_time = prices["open_time"].values[i]
        hour_utc = (open_time // 3600000) % 24  # Convert ms to hours
        
        in_session = (hour_utc >= 6) and (hour_utc <= 22)
        
        if not in_session:
            # Close any open positions outside session
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4H TREND BIAS (PRIMARY) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15M TREND CONFIRMATION ===
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        # === 5M LOCAL TREND ===
        hma_5m_bull = close[i] > hma_5m[i]
        hma_5m_bear = close[i] < hma_5m[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 1e-10 else 0.0
        vol_confirmed = vol_ratio > 1.2
        
        # === RSI PULLBACK CONDITIONS ===
        # Long: RSI pulled back but still bullish (35-55 range in uptrend)
        rsi_pullback_long = (rsi[i] >= 35.0) and (rsi[i] <= 55.0)
        rsi_pullback_short = (rsi[i] >= 45.0) and (rsi[i] <= 65.0)
        
        # RSI turning up/down
        rsi_turning_up = (i > 0) and (not np.isnan(rsi[i-1])) and (rsi[i] > rsi[i-1])
        rsi_turning_down = (i > 0) and (not np.isnan(rsi[i-1])) and (rsi[i] < rsi[i-1])
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m bull + RSI pullback + volume + price above SMA50
        if htf_4h_bull and htf_15m_bull and rsi_pullback_long and rsi_turning_up:
            if above_sma50 and vol_confirmed:
                desired_signal = SIZE
        
        # SHORT: 4h bear + 15m bear + RSI pullback + volume + price below SMA50
        elif htf_4h_bear and htf_15m_bear and rsi_pullback_short and rsi_turning_down:
            if below_sma50 and vol_confirmed:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
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
        
        signals[i] = final_signal
    
    return signals