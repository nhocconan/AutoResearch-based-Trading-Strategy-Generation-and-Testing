#!/usr/bin/env python3
"""
Experiment #121: 15m Primary + 4h/1d HTF — CPR Breakout + RSI Pullback + Session Filter

Hypothesis: 15m strategies fail due to either too many trades (fee drag) or too few (0 trades).
Solution: Use 4h HMA for trend bias (loose filter), 1d CPR for key levels, 15m RSI(7) for timing.
- Session filter: UTC 00-12 (London/NY overlap = higher volume, cleaner moves)
- Entry: Pullback to CPR level + RSI(7) oversold/overbought + HTF trend alignment
- Target: 50-80 trades/year (strict enough to avoid fee drag, loose enough to generate trades)
- Position size: 0.18 (18% of capital, conservative for 15m frequency)

Key design choices:
- Timeframe: 15m (primary), 4h/1d (HTF bias)
- CPR (Central Pivot Range): Previous day's (H+L+2C)/4 for pivot, TC/BC for breakout levels
- RSI(7): Fast RSI for 15m entries (more responsive than RSI(14))
- Session: Only trade UTC 00-12 (avoids low-volume Asia session whipsaw)
- Stoploss: 2.0x ATR trailing (tighter for 15m swings)
- LOOSE filters to ensure >=30 trades on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_rsi7_session_4h1d_v1"
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

def calculate_cpr_levels(df_1d):
    """
    Calculate CPR (Central Pivot Range) levels from daily data
    Pivot = (High + Low + 2*Close) / 4
    TC (Top Central) = (High + Low) / 2
    BC (Bottom Central) = Pivot
    Returns arrays aligned to 1d bars
    """
    n = len(df_1d)
    pivot = np.zeros(n)
    tc = np.zeros(n)
    bc = np.zeros(n)
    
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    for i in range(n):
        pivot[i] = (high[i] + low[i] + 2.0 * close[i]) / 4.0
        tc[i] = (high[i] + low[i]) / 2.0
        bc[i] = pivot[i]
    
    return pivot, tc, bc

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000.0
    hour = (ts_seconds % 86400) / 3600.0
    return int(hour)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d CPR levels
    pivot_1d, tc_1d, bc_1d = calculate_cpr_levels(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=13)
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (conservative for 15m frequency)
    
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
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 only) ===
        hour = get_session_hour(open_time[i])
        in_session = (hour >= 0 and hour < 12)  # London/NY overlap window
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === CPR LEVELS (from 1d) ===
        # Use previous day's CPR (already shifted by align_htf_to_ltf)
        prev_pivot = pivot_aligned[i]
        prev_tc = tc_aligned[i]
        prev_bc = bc_aligned[i]
        
        # Price position relative to CPR
        price_above_tc = close[i] > prev_tc
        price_below_bc = close[i] < prev_bc
        price_in_cpr = (close[i] >= prev_bc) and (close[i] <= prev_tc)
        
        # === RSI SIGNALS (Fast RSI(7) for 15m) ===
        rsi_oversold = rsi_7[i] < 35.0  # Loose for more trades
        rsi_overbought = rsi_7[i] > 65.0  # Loose for more trades
        rsi_neutral = (rsi_7[i] >= 35.0) and (rsi_7[i] <= 65.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: HTF bull + price pullback to CPR support + RSI oversold
        if htf_bull and hma_bull:
            # Pullback to BC (bottom central) support
            if price_below_bc or (close[i] < prev_pivot and close[i-1] >= prev_pivot):
                if rsi_oversold:
                    desired_signal = SIZE
            # Breakout above TC with momentum
            elif price_above_tc and close[i-1] <= prev_tc:
                if rsi_7[i] > 50.0:  # Momentum confirmation
                    desired_signal = SIZE * 0.8
            # Simple trend continuation in session
            elif in_session and rsi_neutral and close[i] > hma_15m[i]:
                if rsi_7[i] > 45.0 and rsi_7[i] < 60.0:
                    desired_signal = SIZE * 0.5
        
        # SHORT ENTRY: HTF bear + price rally to CPR resistance + RSI overbought
        elif htf_bear and hma_bear:
            # Rally to TC (top central) resistance
            if price_above_tc or (close[i] > prev_pivot and close[i-1] <= prev_pivot):
                if rsi_overbought:
                    desired_signal = -SIZE
            # Breakdown below BC with momentum
            elif price_below_bc and close[i-1] >= prev_bc:
                if rsi_7[i] < 50.0:  # Momentum confirmation
                    desired_signal = -SIZE * 0.8
            # Simple trend continuation in session
            elif in_session and rsi_neutral and close[i] < hma_15m[i]:
                if rsi_7[i] > 40.0 and rsi_7[i] < 55.0:
                    desired_signal = -SIZE * 0.5
        
        # === SESSION FILTER ENFORCEMENT ===
        # Only allow new entries in session, but let positions run outside
        if not in_session and not in_position:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
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
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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