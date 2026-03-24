#!/usr/bin/env python3
"""
Experiment #529: 15m Primary + 1h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m timeframe with strict confluence filters can capture intraday trends
while avoiding fee drag. Key insight from failed #521/#525: entry conditions were TOO
STRICT (0 trades). This version LOOSENS entries while adding volume/session filters
to control trade frequency.

Key differences from failed 15m attempts:
1. LOOSER RSI thresholds (30/70 instead of 25/75) to ensure trades trigger
2. Volume confirmation filter (must be > 0.8x average) to avoid fake breakouts
3. Session filter (00-12 UTC) for crypto liquidity peaks
4. Dual HTF bias: 1d HMA for macro + 1h HMA for medium trend
5. Faster RSI(7) for 15m entries (vs RSI14 which is too slow)
6. ATR(14)*2.0 stoploss (tighter than 2.5x for lower TF)

Strategy logic:
1. 1d HMA(21) = macro trend bias (only trade in direction)
2. 1h HMA(21) = medium trend confirmation
3. 15m HMA(9) vs HMA(21) = fast trend signal
4. RSI(7) pullback = entry timing (30-40 for long, 60-70 for short)
5. Volume > 0.8x 20-bar avg = confirmation
6. Session 00-12 UTC = liquidity filter
7. ATR(14)*2.0 stoploss on all positions

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 15m
Position size: 0.20 (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_vol_1h1d_v1"
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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1h HMA for medium trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate 15m indicators
    hma_9 = calculate_hma(close, period=9)
    hma_21 = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(hma_9[i]) or np.isnan(hma_21[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC for crypto liquidity) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        is_liquid_session = 0 <= hour_utc <= 12
        
        # === VOLUME FILTER (must be > 0.8x average) ===
        vol_ratio = volume[i] / vol_sma[i]
        has_volume = vol_ratio > 0.8
        
        # === HTF BIAS (1d macro + 1h medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1h_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = hma_9[i] > hma_21[i]
        hma_bear = hma_9[i] < hma_21[i]
        
        # HMA slope confirmation
        hma_slope_bull = hma_9[i] > hma_9[i-3] if i >= 3 and not np.isnan(hma_9[i-3]) else False
        hma_slope_bear = hma_9[i] < hma_9[i-3] if i >= 3 and not np.isnan(hma_9[i-3]) else False
        
        # === RSI PULLBACK (LOOSE thresholds to ensure trades) ===
        rsi_pullback_long = 30.0 <= rsi[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi[i] <= 70.0
        rsi_recovery_long = rsi[i] > 35.0 and rsi[i] > rsi[i-1] if i > 0 else False
        rsi_recovery_short = rsi[i] < 65.0 and rsi[i] < rsi[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + HMA bull + RSI pullback + volume + session
        if htf_bull and hma_bull:
            confluence_count = 0
            if rsi_pullback_long:
                confluence_count += 1
            if rsi_recovery_long:
                confluence_count += 1
            if has_volume:
                confluence_count += 1
            if is_liquid_session:
                confluence_count += 1
            if hma_slope_bull:
                confluence_count += 1
            
            if confluence_count >= 3:
                desired_signal = SIZE_STRONG if confluence_count >= 4 else SIZE_BASE
        
        # SHORT: HTF bear + HMA bear + RSI pullback + volume + session
        elif htf_bear and hma_bear:
            confluence_count = 0
            if rsi_pullback_short:
                confluence_count += 1
            if rsi_recovery_short:
                confluence_count += 1
            if has_volume:
                confluence_count += 1
            if is_liquid_session:
                confluence_count += 1
            if hma_slope_bear:
                confluence_count += 1
            
            if confluence_count >= 3:
                desired_signal = -SIZE_STRONG if confluence_count >= 4 else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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