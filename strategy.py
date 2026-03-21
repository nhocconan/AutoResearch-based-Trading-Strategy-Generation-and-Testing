#!/usr/bin/env python3
"""
EXPERIMENT #036 - HMA Trend + RSI Pullback + MACD Momentum + ATR Dynamic Sizing
====================================================================================
Hypothesis: Replace ADX with MACD histogram for momentum confirmation. MACD provides
better entry timing than ADX (which is lagging). Use 4h trend + 15m entries for faster
reaction than 1h. Fix all read-only array issues by creating proper copies.

Key changes from #035:
- MACD histogram instead of ADX for momentum (faster signal)
- 4h trend + 15m entry timeframe (faster than 1h entries)
- Fix all read-only array issues with proper .copy() throughout
- Cleaner state machine without modifying arrays in place
- Test position size 0.30 instead of 0.35 (more conservative)
- ATR stoploss at 2.5*ATR (wider to avoid premature stops)

Why this might beat Sharpe=11.523:
- MACD histogram crosses zero faster than ADX crosses 25
- 15m entries catch pullbacks earlier than 1h
- Proper array handling prevents crashes
- Conservative sizing (0.30 max) controls drawdown better
- Wider stops (2.5*ATR) reduce whipsaw exits
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_macd_atr_15m_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period=16):
    """
    Calculate Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, window):
        result = np.zeros(len(data))
        weights = np.arange(1, window + 1, dtype=np.float64)
        for i in range(window - 1, len(data)):
            result[i] = np.sum(data[i - window + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    hma[:period] = close[:period]
    
    return hma


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period + 1])
    
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    mask = (avg_loss > 0) & (avg_gain >= 0)
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[avg_loss == 0] = 100
    
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    # Create copies of all price arrays to avoid read-only issues
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy() if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    if n < 100:
        return np.zeros(n)
    
    # 15m indicators for entry timing and risk
    rsi_15m = calculate_rsi(close, period=14)
    atr_15m = calculate_atr(high, low, close, period=14)
    hma_16_15m = calculate_hma(close, period=16)
    hma_48_15m = calculate_hma(close, period=48)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # 4h HMA for trend filter (resample 15m → 4h)
    df_15m = pd.DataFrame({
        'open': close.copy(),
        'high': high.copy(),
        'low': low.copy(),
        'close': close.copy(),
        'volume': volume.copy()
    })
    df_15m.index = pd.date_range(start='2021-01-01', periods=n, freq='15min')
    
    # Resample to 4h (16 x 15m = 4h)
    df_4h = df_15m.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values.copy()
    h_4h = df_4h['high'].values.copy()
    l_4h = df_4h['low'].values.copy()
    n_4h = len(c_4h)
    
    if n_4h < 50:
        return np.zeros(n)
    
    # Calculate 4h HMA for trend
    hma_16_4h = calculate_hma(c_4h, period=16)
    hma_48_4h = calculate_hma(c_4h, period=48)
    
    # 4h trend direction based on HMA cross and price position
    trend_4h = np.zeros(n_4h)
    for i in range(48, n_4h):
        if hma_16_4h[i] > hma_48_4h[i] and c_4h[i] > hma_16_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif hma_16_4h[i] < hma_48_4h[i] and c_4h[i] < hma_16_4h[i]:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 15m timeframe (16 x 15m = 4h)
    trend_15m = np.zeros(n)
    idx_15m_to_4h = np.arange(n) // 16
    
    for i in range(n):
        idx_4h = idx_15m_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_15m[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.30   # Full position (conservative)
    SIZE_HALF = 0.15   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # MACD histogram threshold for momentum confirmation
    MACD_MIN = 0.0        # MACD histogram must be positive for longs
    
    # ATR stoploss multiplier (wider to avoid whipsaws)
    ATR_STOP_MULT = 2.5   # 2.5*ATR stoploss
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02  # Target 2% ATR
    
    first_valid = max(50, 48, 14, 35)  # Wait for all indicators
    
    # Track position state (local variables, not array modifications)
    in_position = False
    position_side = 0  # 1 for long, -1 for short, 0 for flat
    entry_price = 0.0
    tp_triggered = False
    trailing_stop_price = 0.0
    entry_atr = 0.0
    
    for i in range(first_valid, n):
        # Check for NaN or zero values
        if np.isnan(rsi_15m[i]) or np.isnan(atr_15m[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            entry_price = 0.0
            tp_triggered = False
            trailing_stop_price = 0.0
            continue
        
        trend = trend_15m[i]
        rsi_val = rsi_15m[i]
        macd_val = macd_hist[i]
        atr = atr_15m[i]
        price = close[i]
        
        # Check trailing stop and take profit for existing positions FIRST
        if in_position:
            if position_side == 1:
                # Update trailing stop (move up only)
                if trailing_stop_price > 0:
                    new_trail = max(trailing_stop_price, entry_price + ATR_STOP_MULT * entry_atr)
                else:
                    new_trail = entry_price - ATR_STOP_MULT * entry_atr
                trailing_stop_price = new_trail
                
                # Stoploss check
                if price < trailing_stop_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    trailing_stop_price = 0.0
                    entry_atr = 0.0
                    continue
                
                # Take profit check (2R)
                tp_price = entry_price + TP_MULT * ATR_STOP_MULT * entry_atr
                if not tp_triggered and price >= tp_price:
                    signals[i] = SIZE_HALF
                    tp_triggered = True
                    trailing_stop_price = entry_price  # Trail to breakeven
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    trailing_stop_price = 0.0
                    entry_atr = 0.0
                    continue
                    
            elif position_side == -1:
                # Update trailing stop (move down only)
                if trailing_stop_price > 0:
                    new_trail = min(trailing_stop_price, entry_price - ATR_STOP_MULT * entry_atr)
                else:
                    new_trail = entry_price + ATR_STOP_MULT * entry_atr
                trailing_stop_price = new_trail
                
                # Stoploss check
                if price > trailing_stop_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    trailing_stop_price = 0.0
                    entry_atr = 0.0
                    continue
                
                # Take profit check (2R)
                tp_price = entry_price - TP_MULT * ATR_STOP_MULT * entry_atr
                if not tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    tp_triggered = True
                    trailing_stop_price = entry_price  # Trail to breakeven
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    trailing_stop_price = 0.0
                    entry_atr = 0.0
                    continue
            
            # Hold position - maintain current signal
            signals[i] = signals[i - 1] if i > 0 else 0.0
            continue
        
        # No position - check for new entry signals
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0.02
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        position_size = SIZE_FULL * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_HALF, position_size))  # Clamp
        
        if trend == 1:  # 4h uptrend
            # RSI pullback entry + MACD momentum confirmation
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30 and macd_val > MACD_MIN:
                signals[i] = position_size
                in_position = True
                position_side = 1
                entry_price = price
                tp_triggered = False
                trailing_stop_price = price - ATR_STOP_MULT * atr
                entry_atr = atr
            else:
                signals[i] = 0.0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry + MACD momentum confirmation
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70 and macd_val < -MACD_MIN:
                signals[i] = -position_size
                in_position = True
                position_side = -1
                entry_price = price
                tp_triggered = False
                trailing_stop_price = price + ATR_STOP_MULT * atr
                entry_atr = atr
            else:
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals