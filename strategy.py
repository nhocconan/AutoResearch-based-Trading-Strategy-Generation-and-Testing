#!/usr/bin/env python3
"""
Experiment #613: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe can work with EXTREME selectivity using HTF trend filter.
Key insight from failures: 5m strategies fail due to (1) too many trades = fee drag,
or (2) too strict filters = 0 trades. This strategy balances both.

Strategy logic:
1. 4h HMA(21) = macro trend bias (ONLY trade in this direction)
2. 15m HMA(21) = medium trend confirmation (must align with 4h)
3. 5m RSI(14) = entry timing on pullbacks (RSI 35-45 long, 55-65 short)
4. Session filter: 08-20 UTC only (major market hours, avoid Asia low liquidity)
5. Volume filter: volume > 0.8 * volume_sma(20) (avoid dead periods)
6. ATR(14)*2.5 stoploss on all positions

Key differences from failed 5m/15m attempts:
- RSI pullback bands WIDER (35-45 / 55-65) to ensure trades generate
- Volume filter relaxed (0.8x instead of 1.2x) to avoid 0 trades
- Session filter kept (MANDATORY for 5m per rules) but not overly restrictive
- Position size smaller (0.15 base, 0.20 strong) to handle fee drag

Target: 60-120 trades/year, Sharpe>0.40, DD<-30%
Timeframe: 5m
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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    volume_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return volume_sma

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
    
    # Calculate and align 15m HMA for medium trend
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    # Calculate and align 4h HMA for macro trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 5m indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        
        if np.isnan(hma_15m_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === HTF TREND BIAS (4h + 15m must align) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_15m_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_15m_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === RSI PULLBACK ZONES ===
        # Long: RSI pulled back to 35-45 in uptrend
        rsi_long_pullback = 35.0 <= rsi[i] <= 48.0
        # Short: RSI pulled back to 55-65 in downtrend
        rsi_short_pullback = 52.0 <= rsi[i] <= 65.0
        # RSI recovering from oversold
        rsi_long_recovery = rsi[i] < 40.0 and i > 0 and rsi[i] > rsi[i-1]
        # RSI falling from overbought
        rsi_short_recovery = rsi[i] > 60.0 and i > 0 and rsi[i] < rsi[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries (only in bull trend, in session, volume ok)
        if htf_bull and in_session and vol_ok:
            if rsi_long_pullback:
                desired_signal = SIZE_BASE
            elif rsi_long_recovery:
                desired_signal = SIZE_BASE * 0.8
        
        # SHORT entries (only in bear trend, in session, volume ok)
        if htf_bear and in_session and vol_ok:
            if rsi_short_pullback:
                desired_signal = -SIZE_BASE
            elif rsi_short_recovery:
                desired_signal = -SIZE_BASE * 0.8
        
        # Strong signal when RSI is at extreme of pullback zone
        if htf_bull and in_session and vol_ok and 35.0 <= rsi[i] <= 40.0:
            desired_signal = SIZE_STRONG
        if htf_bear and in_session and vol_ok and 60.0 <= rsi[i] <= 65.0:
            desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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