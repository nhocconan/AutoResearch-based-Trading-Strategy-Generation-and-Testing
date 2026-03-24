#!/usr/bin/env python3
"""
Experiment #157: 15m Primary + 4h/12h HTF — HMA Trend + RSI Mean Reversion + Session Filter

Hypothesis: 15m timeframe has ZERO successful experiments (all Sharpe=0.000 = no trades).
The key is using HTF (4h/12h) for DIRECTION and 15m only for ENTRY TIMING.
This gives HTF trade frequency (~50-100/year) with 15m execution precision.

Key design decisions:
- 12h HMA(50) for major trend bias (slow, stable direction filter)
- 4h HMA(21) for intermediate trend confirmation
- 15m RSI(7) for entry timing (oversold <25 long, overbought >75 short)
- Bollinger Band position: enter only when price at band extremes (mean reversion)
- Session filter: only trade 00-12 UTC (London+NY overlap = high volume)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.20 (smaller for 15m frequency, reduces fee drag impact)

Why this might work where others failed:
- Previous 15m strategies had ZERO trades (conditions too strict OR too loose)
- This uses LOOSE RSI (7-period, extremes 25/75) to ensure entries
- HTF filters ensure we only trade WITH the major trend
- Session filter reduces trades to ~50-100/year (critical for 15m fee management)
- BB position filter ensures we enter at mean-reversion extremes

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_bb_session_4h12h_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands - for mean reversion entry timing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_bb_position(close, upper, lower):
    """
    Bollinger Band Position: where is price relative to bands?
    0.0 = at lower band, 0.5 = at middle (SMA), 1.0 = at upper band
    """
    n = len(close)
    bb_pos = np.zeros(n)
    bb_pos[:] = np.nan
    
    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        band_width = upper[i] - lower[i]
        if band_width > 1e-10:
            bb_pos[i] = (close[i] - lower[i]) / band_width
        else:
            bb_pos[i] = 0.5
    
    return bb_pos

def is_session_active(open_time):
    """
    Session filter: only trade 00-12 UTC (London + NY overlap for crypto)
    open_time is in milliseconds since epoch
    """
    # Convert to hours UTC
    hour_utc = (open_time // 3600000) % 24
    return 0 <= hour_utc < 12

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=7)  # Fast RSI for entry timing
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_sma, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_pos = calculate_bb_position(close, bb_upper, bb_lower)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (smaller for 15m frequency)
    
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
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_pos[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        session_ok = is_session_active(open_time[i])
        
        # === HTF BIAS (12h HMA - major trend) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        itf_bull = close[i] > hma_4h_aligned[i]
        itf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI EXTREMES (Fast RSI(7) for mean reversion) ===
        rsi_oversold = rsi[i] < 25.0  # Very oversold
        rsi_overbought = rsi[i] > 75.0  # Very overbought
        
        # === BOLLINGER BAND POSITION ===
        # bb_pos < 0.15 = near lower band (long opportunity)
        # bb_pos > 0.85 = near upper band (short opportunity)
        bb_long_setup = bb_pos[i] < 0.15
        bb_short_setup = bb_pos[i] > 0.85
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: HTF bull + ITF bull + RSI oversold + BB at lower band + session active
        if htf_bull and itf_bull and rsi_oversold and bb_long_setup and session_ok:
            desired_signal = SIZE
        
        # SHORT: HTF bear + ITF bear + RSI overbought + BB at upper band + session active
        elif htf_bear and itf_bear and rsi_overbought and bb_short_setup and session_ok:
            desired_signal = -SIZE
        
        # FALLBACK LONG: Strong HTF alignment (ignore BB if RSI very extreme)
        elif htf_bull and itf_bull and rsi[i] < 20.0 and session_ok:
            desired_signal = SIZE * 0.8
        
        # FALLBACK SHORT: Strong HTF alignment (ignore BB if RSI very extreme)
        elif htf_bear and itf_bear and rsi[i] > 80.0 and session_ok:
            desired_signal = -SIZE * 0.8
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.6:
            final_signal = SIZE * 0.8
        elif desired_signal <= -SIZE * 0.6:
            final_signal = -SIZE * 0.8
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