#!/usr/bin/env python3
"""
EXPERIMENT #003 - MTF EMA Donchian MACD ADX (15m+4h)
==================================================================================================
Hypothesis: Combine 4H EMA(21/55) crossover for trend + 4H Donchian breakout confirmation
with 15m MACD histogram entry timing + ADX strength filter.
This differs from #001/#002 (Supertrend+MACD+RSI) by using:
- EMA crossover instead of Supertrend for smoother trend detection
- Donchian channel breakout as additional trend confirmation
- ADX filter to avoid weak trending conditions
- MACD histogram cross (not just signal line cross) for better entry timing

Key parameters:
- Position size: 0.30 (conservative, reduces DD)
- Stoploss: 2.0*ATR
- Take profit: 2R then trail at 1R
- ADX min: 25 (only trade when trend has strength)
- Discrete signal levels to reduce churning costs
"""

import numpy as np
import pandas as pd

name = "mtf_ema_donchian_macd_adx_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    multiplier = 2.0 / (period + 1)
    
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)"""
    n = len(close) if 'close' in dir() else len(high)
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    signal_line[slow + signal - 1] = np.mean(macd_line[slow:slow + signal])
    
    multiplier = 2.0 / (signal + 1)
    for i in range(slow + signal, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
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
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    plus_di[period - 1] = 100 * np.sum(plus_dm[1:period]) / np.sum(tr[1:period]) if np.sum(tr[1:period]) > 0 else 0
    minus_di[period - 1] = 100 * np.sum(minus_dm[1:period]) / np.sum(tr[1:period]) if np.sum(tr[1:period]) > 0 else 0
    
    if plus_di[period - 1] + minus_di[period - 1] > 0:
        dx[period - 1] = 100 * abs(plus_di[period - 1] - minus_di[period - 1]) / (plus_di[period - 1] + minus_di[period - 1])
    else:
        dx[period - 1] = 0
    
    adx[period * 2 - 1] = np.mean(dx[period - 1:period * 2])
    
    for i in range(period * 2, n):
        plus_di[i] = 100 * ((plus_di[i - 1] * (period - 1) + plus_dm[i]) / period) / (
            ((np.sum(tr[i - period + 1:i + 1]) / period)) if np.sum(tr[i - period + 1:i + 1]) > 0 else 1
        )
        minus_di[i] = 100 * ((minus_di[i - 1] * (period - 1) + minus_dm[i]) / period) / (
            ((np.sum(tr[i - period + 1:i + 1]) / period)) if np.sum(tr[i - period + 1:i + 1]) > 0 else 1
        )
        
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
        
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def resample_to_higher_tf(prices, target_tf='4h'):
    """Resample to higher timeframe using open_time index - CRITICAL for no look-ahead"""
    prices_indexed = prices.set_index('open_time')
    df_resampled = prices_indexed.resample(target_tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return df_resampled


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    macd_15m, signal_15m, hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    rsi_15m = calculate_rsi(close, period=14)
    
    # Resample to 4h for trend filters using proper method
    try:
        df_4h = resample_to_higher_tf(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        n_4h = len(c_4h)
        
        # 4h indicators for trend
        ema21_4h = calculate_ema(c_4h, 21)
        ema55_4h = calculate_ema(c_4h, 55)
        donchian_upper_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        # Map 4h indicators back to 15m timeframe using reindex with shift(1)
        prices_indexed = prices.set_index('open_time')
        df_4h_indexed = df_4h.copy()
        
        # CRITICAL: shift by 1 to only use COMPLETED 4h bars (avoid look-ahead!)
        df_4h_shifted = df_4h_indexed.shift(1)
        
        # Reindex with forward fill to align 4h data to 15m timestamps
        trend_4h_aligned = df_4h_shifted['close'].reindex(prices_indexed.index, method='ffill').values
        ema21_aligned = df_4h_shifted['close'].reindex(prices_indexed.index, method='ffill').values
        ema55_aligned = df_4h_shifted['close'].reindex(prices_indexed.index, method='ffill').values
        adx_aligned = df_4h_shifted['close'].reindex(prices_indexed.index, method='ffill').values
        donchian_u_aligned = df_4h_shifted['high'].reindex(prices_indexed.index, method='ffill').values
        donchian_l_aligned = df_4h_shifted['low'].reindex(prices_indexed.index, method='ffill').values
        
        # Recompute aligned indicators properly
        trend_4h = np.zeros(n)
        adx_4h_mapped = np.zeros(n)
        donchian_breakout = np.zeros(n)
        
        for i in range(n):
            ts = prices_indexed.index[i]
            mask = df_4h_indexed.index <= ts
            if mask.sum() > 1:  # Need at least 2 completed 4h bars for shift
                idx_4h = mask.sum() - 2  # -2 because we shift by 1
                if idx_4h >= 55:  # Need enough data for EMA55
                    if ema21_4h[idx_4h] > ema55_4h[idx_4h]:
                        trend_4h[i] = 1
                    elif ema21_4h[idx_4h] < ema55_4h[idx_4h]:
                        trend_4h[i] = -1
                    
                    adx_4h_mapped[i] = adx_4h[idx_4h]
                    
                    # Donchian breakout confirmation
                    if c_4h[idx_4h] > donchian_upper_4h[idx_4h - 1] if idx_4h > 0 else c_4h[idx_4h]:
                        donchian_breakout[i] = 1
                    elif c_4h[idx_4h] < donchian_lower_4h[idx_4h - 1] if idx_4h > 0 else c_4h[idx_4h]:
                        donchian_breakout[i] = -1
    except Exception:
        # Fallback: simple bar counting method
        bars_per_4h = 16
        n_4h = n // bars_per_4h
        
        c_4h = np.zeros(n_4h)
        h_4h = np.zeros(n_4h)
        l_4h = np.zeros(n_4h)
        
        for i in range(n_4h):
            start_idx = i * bars_per_4h
            end_idx = min(start_idx + bars_per_4h, n)
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
        
        ema21_4h = calculate_ema(c_4h, 21)
        ema55_4h = calculate_ema(c_4h, 55)
        donchian_upper_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        trend_4h = np.zeros(n)
        adx_4h_mapped = np.zeros(n)
        donchian_breakout = np.zeros(n)
        
        for i in range(n):
            idx_4h = max(0, i // bars_per_4h - 1)  # Shift by 1 bar
            if idx_4h >= 55:
                if ema21_4h[idx_4h] > ema55_4h[idx_4h]:
                    trend_4h[i] = 1
                elif ema21_4h[idx_4h] < ema55_4h[idx_4h]:
                    trend_4h[i] = -1
                
                adx_4h_mapped[i] = adx_4h[idx_4h]
                
                if idx_4h > 0:
                    if c_4h[idx_4h] > donchian_upper_4h[idx_4h - 1]:
                        donchian_breakout[i] = 1
                    elif c_4h[idx_4h] < donchian_lower_4h[idx_4h - 1]:
                        donchian_breakout[i] = -1
    
    signals = np.zeros(n)
    
    # Position sizing - conservative to control drawdown
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Entry filters
    ADX_MIN = 25  # Only trade when trend has strength
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    
    # Risk management
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 55 * 16, 14 * 2)
    
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        adx_val = adx_4h_mapped[i]
        donchian = donchian_breakout[i]
        rsi_val = rsi_15m[i]
        macd_hist = hist_15m[i]
        macd_hist_prev = hist_15m[i - 1] if i > 0 else 0
        atr = atr_15m[i]
        price = close[i]
        
        # ADX filter - only trade when trend has strength
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend and Donchian must agree
        if trend != donchian or trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Handle existing position
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # New entry logic
        if trend == 1 and donchian == 1:
            # Long entry: MACD histogram crossing above zero, RSI in pullback zone
            if (macd_hist > 0 and macd_hist_prev <= 0 and
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and donchian == -1:
            # Short entry: MACD histogram crossing below zero, RSI in pullback zone
            if (macd_hist < 0 and macd_hist_prev >= 0 and
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals