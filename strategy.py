#!/usr/bin/env python3
"""
Experiment #153: 5m Primary + 15m/4h HTF — Session-Filtered Trend Following

Hypothesis: 5m timeframe is completely unexplored (0 experiments) and offers unique edge
when combined with strict session filters and dual-HTF trend confirmation.

Key insights from 150+ failed experiments:
- Complex regime-switching fails on BTC/ETH
- Strategies with Sharpe=0.000 had ZERO trades (entry too strict)
- Session filter is MANDATORY for 5m (avoid Asian session noise)
- 5m needs smaller position size (0.15-0.20) due to more trades = fee drag

New approach for 5m:
- 4h HMA(21) for major trend bias (HTF level 1)
- 15m HMA(13) for intermediate trend (HTF level 2)
- 5m RSI(7) for fast entry timing (adapted for 5m speed)
- Session filter: 08:00-20:00 UTC only (London/NY overlap = high volume)
- Volume spike confirmation (avoid dead periods)
- ATR(14) 2.5x trailing stop for risk management
- Position size: 0.20 (20% of capital - smaller due to trade frequency)

Design for trade generation:
- LOOSE RSI thresholds (long: RSI<55, short: RSI>45) to ensure entries
- Dual HTF alignment (both 15m and 4h must agree)
- Session filter ensures we only trade high-liquidity periods
- Target 50-120 trades/year on 5m timeframe

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_hma_rsi_session_dual_htf_v1"
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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    
    return vol_ratio

def is_session_active(open_time):
    """
    Check if timestamp is within active trading session (08:00-20:00 UTC)
    open_time is in milliseconds since epoch
    """
    # Convert to hours UTC
    ts_seconds = open_time / 1000
    hour_utc = (ts_seconds % 86400) / 3600
    
    # Active session: 08:00 to 20:00 UTC (London/NY overlap)
    return 8.0 <= hour_utc <= 20.0

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
    
    # Calculate and align 15m HMA for intermediate trend
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=13)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    # Calculate and align 4h HMA for major trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (5m) indicators
    hma_5m = calculate_hma(close, period=9)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 5m
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (smaller for 5m due to trade frequency)
    
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
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        session_active = is_session_active(open_time[i])
        
        # === VOLUME CONFIRMATION ===
        volume_ok = vol_ratio[i] > 0.7  # At least 70% of avg volume
        
        # === HTF BIAS (4h HMA - Major Trend) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF BIAS (15m HMA - Intermediate Trend) ===
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        # === DUAL HTF ALIGNMENT (both must agree) ===
        htf_aligned_bull = htf_4h_bull and htf_15m_bull
        htf_aligned_bear = htf_4h_bear and htf_15m_bear
        
        # === 5m HMA TREND ===
        hma_bull = close[i] > hma_5m[i]
        hma_bear = close[i] < hma_5m[i]
        
        # === RSI ENTRY (LOOSE thresholds to ensure trades) ===
        # For 5m, RSI(7) moves faster, so use wider thresholds
        rsi_ok_long = rsi[i] < 55.0  # Not overbought (allow entries up to 55)
        rsi_ok_short = rsi[i] > 45.0  # Not oversold (allow entries down to 45)
        
        # === RSI MOMENTUM CONFIRMATION ===
        rsi_momentum_long = rsi[i] > 35.0  # Not extremely oversold
        rsi_momentum_short = rsi[i] < 65.0  # Not extremely overbought
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY: All conditions aligned (full size)
        if session_active and volume_ok and htf_aligned_bull and hma_bull and rsi_ok_long and rsi_momentum_long:
            desired_signal = SIZE
        
        elif session_active and volume_ok and htf_aligned_bear and hma_bear and rsi_ok_short and rsi_momentum_short:
            desired_signal = -SIZE
        
        # FALLBACK: Strong HTF alignment (ignore 5m HMA if HTF very strong) - 70% size
        elif session_active and volume_ok and htf_aligned_bull and rsi[i] > 40.0 and rsi[i] < 50.0:
            desired_signal = SIZE * 0.7
        
        elif session_active and volume_ok and htf_aligned_bear and rsi[i] > 50.0 and rsi[i] < 60.0:
            desired_signal = -SIZE * 0.7
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
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
    
    return signals