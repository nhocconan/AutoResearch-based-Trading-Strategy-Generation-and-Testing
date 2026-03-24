#!/usr/bin/env python3
"""
Experiment #169: 15m Primary + 1h/4h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m timeframe is underexplored (ZERO experiments in history). Previous 15m
attempts (#157, #159, #161, #165) all failed with Sharpe=0.000 = ZERO trades due to
overly strict confluence filters.

Key learnings from failed 15m experiments:
- Too many HTF filters = 0 trades (all 4 previous 15m attempts failed this way)
- Session filters can kill trade generation if too strict
- RSI thresholds must be LOOSE (40/60 not 30/70) to ensure trades
- Need fallback entry paths when primary conditions don't trigger

New approach for 15m:
- 1d HMA(50) for major regime bias (only ONE HTF filter, not 3)
- 4h HMA(21) for intermediate trend confirmation
- 15m RSI(7) for entry timing (faster than RSI14)
- LOOSE RSI thresholds (>40 long, <60 short) to ENSURE trades
- Session filter: prefer 00-12 UTC but allow trades outside if HTF strongly aligned
- ATR ratio filter < 2.0 (not 1.8) to allow more entries
- 2.0x ATR trailing stop for risk management
- Position size: 0.20 (smaller for higher frequency)

Design for trade generation (CRITICAL - avoid 0 trades):
- Only 2 HTF filters (1d + 4h), not 3 (1w was killing trades)
- LOOSE RSI thresholds (40/60)
- Session filter is OPTIONAL (fallback entries ignore it)
- Multiple entry paths with decreasing size requirements
- Target 50-100 trades/year on 15m timeframe

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_4h1d_v2"
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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for volatility filter"""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    n = len(close)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major regime bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=16)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (smaller for 15m frequency)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
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
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === Session Filter (00-12 UTC preferred for London/NY overlap) ===
        # Extract hour from open_time (milliseconds since epoch)
        hour_utc = (open_time[i] // 3600000) % 24
        is_preferred_session = 0 <= hour_utc <= 12
        
        # === HTF BIAS (1d HMA) - Major regime ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY FILTER (LOOSE - < 2.0 not 1.8) ===
        vol_ok = atr_ratio[i] < 2.0
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI CONFIRMATION (LOOSE to ensure trades) ===
        rsi_ok_long = rsi[i] > 40.0  # Very loose
        rsi_ok_short = rsi[i] < 60.0  # Very loose
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY: All conditions aligned + preferred session (full size)
        if hma_bull and htf_4h_bull and htf_1d_bull and vol_ok and rsi_ok_long and is_preferred_session:
            desired_signal = SIZE
        
        elif hma_bear and htf_4h_bear and htf_1d_bear and vol_ok and rsi_ok_short and is_preferred_session:
            desired_signal = -SIZE
        
        # FALLBACK 1: Strong HTF alignment (ignore session + vol) - 80% size
        # This ensures trades outside preferred hours when trend is strong
        elif hma_bull and htf_4h_bull and htf_1d_bull and rsi[i] > 45.0:
            desired_signal = SIZE * 0.8
        
        elif hma_bear and htf_4h_bear and htf_1d_bear and rsi[i] < 55.0:
            desired_signal = -SIZE * 0.8
        
        # FALLBACK 2: 4h + 15m aligned (ignore 1d) - 60% size
        # Ensures trades when daily is choppy but 4h trend is clear
        elif hma_bull and htf_4h_bull and vol_ok and rsi[i] > 42.0:
            desired_signal = SIZE * 0.6
        
        elif hma_bear and htf_4h_bear and vol_ok and rsi[i] < 58.0:
            desired_signal = -SIZE * 0.6
        
        # FALLBACK 3: Very strong 15m momentum (ignore HTF) - 40% size
        # Ensures we get SOME trades even in choppy markets
        elif hma_bull and rsi[i] > 50.0 and vol_ok:
            desired_signal = SIZE * 0.4
        
        elif hma_bear and rsi[i] < 50.0 and vol_ok:
            desired_signal = -SIZE * 0.4
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.7:
            final_signal = SIZE * 0.8
        elif desired_signal <= -SIZE * 0.7:
            final_signal = -SIZE * 0.8
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
        elif desired_signal >= SIZE * 0.3:
            final_signal = SIZE * 0.4
        elif desired_signal <= -SIZE * 0.3:
            final_signal = -SIZE * 0.4
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