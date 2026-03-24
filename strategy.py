#!/usr/bin/env python3
"""
Experiment #493: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m has ZERO prior experiments. Success requires:
1. STRICT session filter (08-20 UTC) to avoid Asian session noise
2. 4h HMA(21) for primary trend bias (only trade WITH trend)
3. 15m RSI(7) pullback entries (not 5m RSI - too noisy)
4. Loose RSI thresholds (35/65) to ensure trade generation
5. Small position size (0.15-0.20) due to higher trade frequency

Key differences from failed experiments:
- SESSION FILTER: Only trade 08:00-20:00 UTC (London/NY overlap)
- HTF RSI: Use 15m RSI for entries, not 5m (less noise)
- SINGLE HTF: 4h HMA only (not complex 12h+1d regime)
- LOOSE thresholds: RSI 35/65 not 30/70 to guarantee trades

Target: Sharpe>0.40, trades>=150 train (37/year), trades>=20 test
Timeframe: 5m (unexplored - high potential with proper filtering)
Position Size: 0.15 base, 0.20 strong (small due to fee drag on 5m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_rsi_pullback_4h15m_v1"
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

def get_hour_from_open_time(open_time_array):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    # Convert to hours UTC
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for primary trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 15m RSI for entry timing
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=7)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Also calculate 5m RSI for confirmation (not primary entry)
    rsi_5m = calculate_rsi(close, period=7)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        
        if np.isnan(rsi_5m[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        # Only trade 08:00-20:00 UTC (London/NY overlap, avoid Asian noise)
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4h HTF TREND BIAS (PRIMARY FILTER) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === 15m RSI PULLBACK (LOOSE: 35/65) ===
        rsi_15m_val = rsi_15m_aligned[i]
        rsi_15m_oversold = rsi_15m_val < 35.0
        rsi_15m_overbought = rsi_15m_val > 65.0
        rsi_15m_extreme_oversold = rsi_15m_val < 30.0
        rsi_15m_extreme_overbought = rsi_15m_val > 70.0
        
        # === 5m RSI CONFIRMATION ===
        rsi_5m_val = rsi_5m[i]
        rsi_5m_oversold = rsi_5m_val < 40.0
        rsi_5m_overbought = rsi_5m_val > 60.0
        
        # === ENTRY LOGIC (TREND + PULLBACK) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m RSI pullback + 5m confirmation
        if htf_bull and above_sma50:
            # Strong long: extreme oversold on 15m
            if rsi_15m_extreme_oversold and rsi_5m_oversold:
                desired_signal = SIZE_STRONG
            # Base long: moderate oversold on 15m
            elif rsi_15m_oversold and rsi_5m_oversold:
                desired_signal = SIZE_BASE
            # Recovery long: 15m RSI crossing above 40
            elif rsi_15m_val > 40.0 and rsi_15m_aligned[i-1] < 40.0 and above_sma200:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 15m RSI pullback + 5m confirmation
        elif htf_bear and below_sma50:
            # Strong short: extreme overbought on 15m
            if rsi_15m_extreme_overbought and rsi_5m_overbought:
                desired_signal = -SIZE_STRONG
            # Base short: moderate overbought on 15m
            elif rsi_15m_overbought and rsi_5m_overbought:
                desired_signal = -SIZE_BASE
            # Weakness short: 15m RSI crossing below 60
            elif rsi_15m_val < 60.0 and rsi_15m_aligned[i-1] > 60.0 and below_sma200:
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
                # New position or flip
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