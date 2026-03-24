#!/usr/bin/env python3
"""
Experiment #061: 15m Primary + 1h/4h HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m strategies fail because entry conditions are too strict (0 trades).
Solution: LOOSE entry conditions with HTF bias, not hard filters.
- 1h HMA(21) provides trend direction (close > HMA = bull bias)
- 4h HMA(50) provides major regime bias (only reduces size, doesn't block)
- 15m RSI(7) for entry timing (loose: <50 long, >50 short - not extreme)
- Session filter: 00-12 UTC gets full size, 12-24 UTC gets 0.7x size (not blocked)
- Donchian(10) breakout for momentum confirmation
- ATR(14) 2.5x trailing stoploss
- Position size: 0.18 (18% - conservative for 15m frequency)

Target: 50-100 trades/year, Sharpe>0.2, DD>-40%, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_donchian_1h4h_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMA for trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=13)
    rsi = calculate_rsi(close, period=7)  # Fast RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=10)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.18  # 18% position size (conservative for 15m)
    
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = full size, 12-24 UTC = reduced) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        is_prime_session = 0 <= hour_utc < 12  # London + NY overlap
        
        session_multiplier = 1.0 if is_prime_session else 0.7
        
        # === HTF BIAS (1h and 4h HMA) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI SIGNAL (LOOSE thresholds to ensure trades) ===
        # Long: RSI < 50 (pullback in uptrend)
        # Short: RSI > 50 (pullback in downtrend)
        rsi_long = rsi[i] < 50.0
        rsi_short = rsi[i] > 50.0
        
        # === DONCHIAN MOMENTUM ===
        # Breakout above previous high = bullish momentum
        # Breakout below previous low = bearish momentum
        donchian_bull = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_bear = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === DESIRED SIGNAL (Multiple entry paths to ensure trades) ===
        desired_signal = 0.0
        signal_strength = 1.0
        
        # PATH 1: Trend pullback entry (most common)
        # Long: 1h bull + 15m HMA bull + RSI pullback
        if htf_1h_bull and hma_bull and rsi_long:
            desired_signal = BASE_SIZE
            signal_strength = 1.0
        # Short: 1h bear + 15m HMA bear + RSI pullback
        elif htf_1h_bear and hma_bear and rsi_short:
            desired_signal = -BASE_SIZE
            signal_strength = 1.0
        
        # PATH 2: Donchian breakout with HTF confirmation
        # Long: Donchian breakout + 1h bull
        elif donchian_bull and htf_1h_bull:
            desired_signal = BASE_SIZE
            signal_strength = 0.9
        # Short: Donchian breakout + 1h bear
        elif donchian_bear and htf_1h_bear:
            desired_signal = -BASE_SIZE
            signal_strength = 0.9
        
        # PATH 3: 4h regime alignment (stronger signal)
        # Long: All three timeframes bull + RSI ok
        if htf_1h_bull and htf_4h_bull and hma_bull and rsi[i] < 55.0:
            desired_signal = BASE_SIZE
            signal_strength = 1.0
        # Short: All three timeframes bear + RSI ok
        elif htf_1h_bear and htf_4h_bear and hma_bear and rsi[i] > 45.0:
            desired_signal = -BASE_SIZE
            signal_strength = 1.0
        
        # PATH 4: Extreme RSI mean reversion (counter-trend, smaller size)
        if rsi[i] < 25.0 and htf_4h_bull:  # Oversold in bull regime
            desired_signal = BASE_SIZE * 0.5
            signal_strength = 0.5
        elif rsi[i] > 75.0 and htf_4h_bear:  # Overbought in bear regime
            desired_signal = -BASE_SIZE * 0.5
            signal_strength = 0.5
        
        # Apply session multiplier
        if desired_signal != 0.0:
            desired_signal = desired_signal * session_multiplier
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.5
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