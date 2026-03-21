#!/usr/bin/env python3
"""
EXPERIMENT #014 - DEMA Trend + CCI Entry + ADX Filter + ATR Stop
================================================================
Hypothesis: DEMA (Double EMA) responds faster to trend changes than regular EMA,
combined with CCI for entry timing within the trend direction. ADX filters out
weak trends to avoid whipsaws. ATR-based trailing stop manages risk dynamically.

Key differences from previous attempts:
- DEMA(21/55) instead of HMA/KAMA for faster trend detection
- CCI(20) for entry timing instead of RSI (different oscillator behavior)
- ADX(14) strength filter to avoid trading in choppy markets
- Cleaner multi-timeframe mapping without artificial date indices
- Discrete signal levels (0, ±0.25, ±0.35) to minimize churn costs

Why this might beat Sharpe=2.139:
- DEMA reduces lag compared to EMA while being smoother than HMA
- CCI captures momentum extremes better than RSI in trending markets
- ADX filter prevents entries during low-trend-strength periods
- Proven multi-timeframe structure from winning strategies
"""

import numpy as np
import pandas as pd

name = "mtf_dema_cci_adx_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    dema = 2 * ema1 - ema2
    dema[:period] = np.nan
    
    return dema


def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index for entry timing"""
    n = len(close)
    tp = (high + low + close) / 3
    
    tp_mean = pd.Series(tp).rolling(window=period, min_periods=period).mean().values
    tp_std = pd.Series(tp).rolling(window=period, min_periods=period).std().values
    
    cci = np.zeros(n)
    mask = tp_std > 0
    cci[mask] = (tp[mask] - tp_mean[mask]) / (0.015 * tp_std[mask])
    
    return cci


def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength filter"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def calculate_atr(high, low, close, period=14):
    """Calculate ATR for trailing stop and position sizing"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    cci_1h = calculate_cci(high, low, close, period=20)
    atr_1h = calculate_atr(high, low, close, period=14)
    adx_1h = calculate_adx(high, low, close, period=14)
    dema_fast_1h = calculate_dema(close, period=21)
    dema_slow_1h = calculate_dema(close, period=55)
    
    # 4h trend via downsampling (integer-based, no date manipulation)
    n_4h = n // 4
    close_4h = close[:n_4h * 4].reshape(n_4h, 4).mean(axis=1)
    high_4h = high[:n_4h * 4].reshape(n_4h, 4).max(axis=1)
    low_4h = low[:n_4h * 4].reshape(n_4h, 4).min(axis=1)
    
    dema_fast_4h = calculate_dema(close_4h, period=21)
    dema_slow_4h = calculate_dema(close_4h, period=55)
    
    # Map 4h trend back to 1h
    trend_4h = np.zeros(n_4h)
    for i in range(55, n_4h):
        if dema_fast_4h[i] > dema_slow_4h[i]:
            trend_4h[i] = 1
        elif dema_fast_4h[i] < dema_slow_4h[i]:
            trend_4h[i] = -1
    
    trend_1h = np.zeros(n)
    for i in range(n):
        idx_4h = i // 4
        if idx_4h < n_4h:
            trend_1h[i] = trend_4h[idx_4h]
    
    # Signal generation
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.35
    SIZE_HALF = 0.25
    
    # Thresholds
    CCI_LONG_ENTRY = -100   # Enter long when CCI oversold in uptrend
    CCI_SHORT_ENTRY = 100   # Enter short when CCI overbought in downtrend
    ADX_MIN = 20            # Minimum ADX for trend strength
    ATR_STOP_MULT = 2.5     # ATR multiplier for trailing stop
    
    # Wait for all indicators to be valid
    first_valid = max(55, 21, 20, 14)
    
    # Track position for stoploss
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(cci_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(dema_fast_1h[i]) or np.isnan(dema_slow_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        cci_val = cci_1h[i]
        adx_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # ADX filter - only trade when trend strength is sufficient
        if adx_val < ADX_MIN:
            if position_side != 0:
                # Check stoploss even in low ADX
                if position_side == 1:
                    stoploss = entry_price - ATR_STOP_MULT * atr
                    if price < stoploss:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        continue
                elif position_side == -1:
                    stoploss = entry_price + ATR_STOP_MULT * atr
                    if price > stoploss:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        continue
                # Hold position
                signals[i] = signals[i - 1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Check trailing stop for existing positions
        if position_side != 0:
            if position_side == 1:  # Long
                if i == 0 or entry_price == 0:
                    entry_price = price
                highest_since_entry = max(highest_since_entry, price)
                stoploss = max(entry_price, highest_since_entry - ATR_STOP_MULT * atr)
                if price < stoploss:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    continue
            elif position_side == -1:  # Short
                if i == 0 or entry_price == 0:
                    entry_price = price
                lowest_since_entry = min(lowest_since_entry, price)
                stoploss = min(entry_price, lowest_since_entry + ATR_STOP_MULT * atr)
                if price > stoploss:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    lowest_since_entry = 0.0
                    continue
        
        # 1h DEMA confirmation
        dema_confirmed_long = dema_fast_1h[i] > dema_slow_1h[i]
        dema_confirmed_short = dema_fast_1h[i] < dema_slow_1h[i]
        
        if trend == 1 and dema_confirmed_long:  # Uptrend
            if cci_val < CCI_LONG_ENTRY:
                signals[i] = SIZE_FULL
                position_side = 1
                entry_price = price
                highest_since_entry = price
            elif cci_val < 0 and position_side == 1:
                signals[i] = SIZE_HALF
            elif position_side == 1:
                signals[i] = SIZE_HALF  # Hold reduced position
            else:
                signals[i] = 0.0
                
        elif trend == -1 and dema_confirmed_short:  # Downtrend
            if cci_val > CCI_SHORT_ENTRY:
                signals[i] = -SIZE_FULL
                position_side = -1
                entry_price = price
                lowest_since_entry = price
            elif cci_val > 0 and position_side == -1:
                signals[i] = -SIZE_HALF
            elif position_side == -1:
                signals[i] = -SIZE_HALF  # Hold reduced position
            else:
                signals[i] = 0.0
        else:
            # No clear trend or DEMA disagreement
            signals[i] = 0.0
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
    
    return signals