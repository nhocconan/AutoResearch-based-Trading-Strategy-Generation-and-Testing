#!/usr/bin/env python3
"""
Experiment #353: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe has ZERO prior experiments. Lower TF strategies typically fail
due to excessive trades → fee drag. This strategy uses EXTREME selectivity:
1. 4h HMA for primary trend bias (only trade in HTF direction)
2. 15m HMA for intermediate confirmation (both HTFs must align)
3. 5m RSI pullback for entry timing (wait for counter-trend exhaustion)
4. Session filter (08-20 UTC) to avoid low-liquidity whipsaws
5. Volume confirmation on entries (1.5x avg volume)

Key differences from failed 15m strategies:
- Stricter HTF alignment (BOTH 4h AND 15m must agree)
- Session filter MANDATORY (no overnight trades)
- Smaller position size (0.15 base) due to higher trade frequency
- RSI thresholds tighter (30/70 instead of 35/65) for fewer entries

Entry Logic:
- Long: 4h HMA bull + 15m HMA bull + 5m RSI < 30 + volume > 1.5x avg + session active
- Short: 4h HMA bear + 15m HMA bear + 5m RSI > 70 + volume > 1.5x avg + session active

Position sizing: 0.15 base, 0.25 when both HTFs strongly aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades=50-120/year, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_hma_rsi_pullback_15m4h_v1"
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

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def is_session_active(open_time_unix_ms):
    """
    Session filter: 08-20 UTC (London/NY overlap + active hours)
    open_time_unix_ms: Binance open_time in milliseconds
    Returns True if within active session
    """
    # Convert ms to hours UTC
    hour_utc = (open_time_unix_ms // 3600000) % 24
    # Active: 08:00 to 20:00 UTC (12 hours)
    return 8 <= hour_utc < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values  # milliseconds since epoch
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMAs for trend bias
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (5m) indicators
    hma_5m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # Trade counter for frequency control
    trades_this_year = 0
    current_year = 0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_5m[i]) or np.isnan(rsi[i]):
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
        
        if np.isnan(sma_200[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        session_active = is_session_active(open_time[i])
        
        if not session_active:
            # Outside session: flatten positions
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
            continue
        
        # === HTF TREND BIAS (BOTH 4h AND 15m must align) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        # Both HTFs must agree for strong signal
        htf_strong_bull = htf_4h_bull and htf_15m_bull
        htf_strong_bear = htf_4h_bear and htf_15m_bear
        
        # At least one HTF for base signal
        htf_any_bull = htf_4h_bull or htf_15m_bull
        htf_any_bear = htf_4h_bear or htf_15m_bear
        
        # === 5m HMA TREND ===
        hma_5m_bull = close[i] > hma_5m[i]
        hma_5m_bear = close[i] < hma_5m[i]
        
        # === SMA200 FILTER (long-term bias) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK (entry timing) ===
        rsi_oversold = rsi[i] < 30.0
        rsi_overbought = rsi[i] > 70.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = False
        if vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        # === TRADE FREQUENCY CONTROL ===
        # Extract year from open_time (milliseconds)
        year = int(open_time[i] // (365 * 24 * 3600000)) + 1970
        if year != current_year:
            trades_this_year = 0
            current_year = year
        
        # Max 120 trades per year on 5m
        max_trades_reached = trades_this_year >= 120
        
        # === ENTRY LOGIC (EXTREME SELECTIVITY) ===
        desired_signal = 0.0
        
        # LONG: Both HTFs bull + RSI oversold + volume confirm + SMA200 filter
        if htf_strong_bull and rsi_oversold and vol_confirm and above_sma200:
            if not max_trades_reached:
                desired_signal = SIZE_STRONG
        elif htf_any_bull and rsi_oversold and vol_confirm:
            if not max_trades_reached:
                desired_signal = SIZE_BASE
        
        # SHORT: Both HTFs bear + RSI overbought + volume confirm + SMA200 filter
        elif htf_strong_bear and rsi_overbought and vol_confirm and below_sma200:
            if not max_trades_reached:
                desired_signal = -SIZE_STRONG
        elif htf_any_bear and rsi_overbought and vol_confirm:
            if not max_trades_reached:
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
                # New position or flip - count trade
                trades_this_year += 1
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