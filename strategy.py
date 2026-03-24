#!/usr/bin/env python3
"""
Experiment #177: 15m Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume

Hypothesis: 15m timeframe with proper HTF filtering can capture intraday momentum
while avoiding whipsaws. Previous 15m attempts failed with 0 trades due to overly
strict conditions. This version SIMPLIFIES entry logic to ensure trade generation:

Core Logic:
- 4h HMA(21) = Major trend direction (ONLY trade in HTF direction)
- 12h HMA(50) = Secondary confirmation (stronger bias when aligned)
- 15m RSI(7) = Entry timing (pullback entries in trend direction)
- Volume filter = Confirm participation (volume > 1.5x 20-bar avg)
- Session filter = Prefer 00-12 UTC (London/NY overlap for crypto liquidity)

Entry Conditions (LOOSENER for trade generation):
- Long: 4h HMA bull + RSI(7) < 40 then crosses above 35 + volume confirm
- Short: 4h HMA bear + RSI(7) > 60 then crosses below 65 + volume confirm

Position Sizing:
- Base: 0.20 (20% of capital)
- Strong (both 4h+12h aligned): 0.25
- Stoploss: 2.5x ATR trailing (signal → 0 when hit)

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test, 40-100 trades/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_vol_session_4h12h_v1"
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
    """Simple Moving Average of Volume"""
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
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for major trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for secondary confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20  # 20% base position size
    SIZE_STRONG = 0.25  # 25% for strong signals (4h+12h aligned)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track RSI cross for entry timing
    prev_rsi = np.nan
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF CONFIRMATION (12h HMA) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === VOLUME FILTER ===
        volume_confirm = volume[i] > 1.5 * vol_sma[i] if vol_sma[i] > 0 else False
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        is_peak_session = 0 <= hour_utc <= 12  # London/NY overlap
        
        # === RSI CROSS DETECTION ===
        rsi_cross_up = False
        rsi_cross_down = False
        
        if not np.isnan(prev_rsi) and not np.isnan(rsi_7[i]):
            # RSI crossed above 35 from below (long entry)
            if prev_rsi < 35.0 and rsi_7[i] >= 35.0:
                rsi_cross_up = True
            # RSI crossed below 65 from above (short entry)
            if prev_rsi > 65.0 and rsi_7[i] <= 65.0:
                rsi_cross_down = True
        
        # === ENTRY LOGIC (SIMPLIFIED for trade generation) ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bull + RSI pullback cross + volume
        if htf_4h_bull and rsi_cross_up:
            # Base signal with volume confirm
            if volume_confirm:
                # Strong signal if 12h also bull
                if htf_12h_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif is_peak_session:
                # Enter on peak session even without volume spike
                desired_signal = SIZE_BASE * 0.8
        
        # SHORT ENTRY: 4h bear + RSI rally cross + volume
        elif htf_4h_bear and rsi_cross_down:
            # Base signal with volume confirm
            if volume_confirm:
                # Strong signal if 12h also bear
                if htf_12h_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif is_peak_session:
                # Enter on peak session even without volume spike
                desired_signal = -SIZE_BASE * 0.8
        
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
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
        prev_rsi = rsi_7[i]
    
    return signals