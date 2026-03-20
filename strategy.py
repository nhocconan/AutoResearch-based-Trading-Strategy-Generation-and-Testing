#!/usr/bin/env python3
"""
EXPERIMENT #012 - Multi-Timeframe Supertrend + MACD + ADX Filter
================================================================
Hypothesis: Combining 4h Supertrend trend filter with 1h MACD histogram 
entry signals + ADX strength filter will reduce whipsaw trades and improve
Sharpe ratio vs HMA+RSI approach.

Key differences from mtf_hma_rsi_zscore_v1:
- Supertrend(ATR=10, mult=3) instead of HMA for trend (built-in volatility stops)
- MACD histogram cross for entry timing (momentum shift vs RSI levels)
- ADX(14) filter to avoid trading in weak/choppy trends
- More selective entries should reduce trade count but improve win rate

Why this might beat Sharpe=5.4:
- Supertrend adapts to volatility better than fixed HMA
- MACD histogram captures momentum shifts earlier than RSI
- ADX filter eliminates low-quality trades in ranging markets
- Fewer but higher quality trades = better risk-adjusted returns
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_macd_adx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=10):
    """Calculate Average True Range with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend values, trend direction (1=up, -1=down)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
            
        upper_band = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            trend[i] = 1 if close[i] > supertrend[i] else -1
        else:
            if trend[i-1] == 1:
                supertrend[i] = max(lower_band, supertrend[i-1])
                if close[i] < supertrend[i]:
                    supertrend[i] = upper_band
                    trend[i] = -1
                else:
                    trend[i] = 1
            else:
                supertrend[i] = min(upper_band, supertrend[i-1])
                if close[i] > supertrend[i]:
                    supertrend[i] = lower_band
                    trend[i] = 1
                else:
                    trend[i] = -1
    
    return supertrend, trend


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD with histogram"""
    n = len(close)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    macd_hist = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_hist


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    plus_di = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values
    minus_di = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    plus_di_pct = np.zeros(n)
    minus_di_pct = np.zeros(n)
    
    mask = atr > 0
    plus_di_pct[mask] = 100 * plus_di[mask] / atr[mask]
    minus_di_pct[mask] = 100 * minus_di[mask] / atr[mask]
    
    dx = np.zeros(n)
    mask2 = (plus_di_pct + minus_di_pct) > 0
    dx[mask2] = 100 * np.abs(plus_di_pct[mask2] - minus_di_pct[mask2]) / (plus_di_pct[mask2] + minus_di_pct[mask2])
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # 4h Supertrend for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Calculate 4h Supertrend
    supertrend_4h, trend_4h = calculate_supertrend(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=10,
        multiplier=3.0
    )
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # MACD histogram thresholds for momentum entry
    MACD_LONG_THRESHOLD = 0.0    # Histogram crossing above zero
    MACD_SHORT_THRESHOLD = 0.0   # Histogram crossing below zero
    
    # ADX threshold for trend strength
    ADX_MIN = 20.0      # Only trade when ADX > 20 (trending market)
    
    # ATR for stoploss tracking
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Track entry prices for stoploss
    entry_price = np.zeros(n)
    position_direction = np.zeros(n)  # 1=long, -1=short, 0=none
    
    first_valid = max(48, 26, 14)  # Wait for all indicators
    
    for i in range(first_valid, n):
        if np.isnan(macd_hist[i]) or np.isnan(adx_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        macd_val = macd_hist[i]
        adx_val = adx_1h[i]
        
        # ADX filter - only trade in trending markets
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            position_direction[i] = 0
            continue
        
        # Check stoploss (2*ATR against position)
        if position_direction[i-1] != 0 and i > 0:
            if position_direction[i-1] == 1:  # Long position
                stop_loss = entry_price[i-1] - 2 * atr_1h[i]
                if close[i] < stop_loss:
                    signals[i] = 0.0
                    position_direction[i] = 0
                    continue
            elif position_direction[i-1] == -1:  # Short position
                stop_loss = entry_price[i-1] + 2 * atr_1h[i]
                if close[i] > stop_loss:
                    signals[i] = 0.0
                    position_direction[i] = 0
                    continue
        
        # Generate new signals based on trend + MACD
        if trend == 1:  # 4h uptrend
            if macd_val > MACD_LONG_THRESHOLD:
                # MACD histogram positive - momentum up
                signals[i] = SIZE_FULL
                if position_direction[i-1] != 1:
                    entry_price[i] = close[i]
                    position_direction[i] = 1
                else:
                    entry_price[i] = entry_price[i-1]
                    position_direction[i] = 1
            elif macd_val > -50:  # Weakening but not reversed
                signals[i] = SIZE_HALF
                position_direction[i] = position_direction[i-1]
                entry_price[i] = entry_price[i-1] if i > 0 else close[i]
            else:
                signals[i] = 0.0
                position_direction[i] = 0
        elif trend == -1:  # 4h downtrend
            if macd_val < MACD_SHORT_THRESHOLD:
                # MACD histogram negative - momentum down
                signals[i] = -SIZE_FULL
                if position_direction[i-1] != -1:
                    entry_price[i] = close[i]
                    position_direction[i] = -1
                else:
                    entry_price[i] = entry_price[i-1]
                    position_direction[i] = -1
            elif macd_val < 50:  # Weakening but not reversed
                signals[i] = -SIZE_HALF
                position_direction[i] = position_direction[i-1]
                entry_price[i] = entry_price[i-1] if i > 0 else close[i]
            else:
                signals[i] = 0.0
                position_direction[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_direction[i] = 0
    
    return signals