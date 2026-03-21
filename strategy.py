#!/usr/bin/env python3
"""
EXPERIMENT #019 - Supertrend + KAMA + MACD Histogram + ADX Filter
====================================================================================
Hypothesis: Replace HMA trend with Supertrend (proven in #007) + KAMA confirmation.
Use MACD histogram crosses for entry timing instead of RSI pullbacks.
Add ADX filter to ensure we only trade strong trends (>25).

Why this might beat Sharpe=5.4:
- Supertrend gives cleaner trend signals with ATR-based stops built-in
- KAMA adapts to volatility better than fixed EMA/HMA
- MACD histogram crosses catch momentum shifts earlier than RSI
- ADX filter avoids choppy markets where most strategies fail
- Different signal combination than current best (HMA+RSI)

Key features:
- 4h Supertrend for primary trend direction
- 4h KAMA(10) for trend confirmation (price vs KAMA)
- 15m MACD histogram cross for entry timing
- 15m ADX(14) > 25 filter for trend strength
- 2*ATR stoploss, 2R take profit (reduce to half)
- Discrete signal levels: 0.0, ±0.25, ±0.35
- Dynamic sizing based on ATR volatility
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_kama_macd_adx_v1"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 for bullish, -1 for bearish
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band = mid + multiplier * atr[i]
        lower_band = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            # Update supertrend based on direction
            if direction[i - 1] == 1:
                supertrend[i] = max(lower_band, supertrend[i - 1])
                if close[i] < supertrend[i]:
                    supertrend[i] = upper_band
                    direction[i] = -1
                else:
                    direction[i] = 1
            else:
                supertrend[i] = min(upper_band, supertrend[i - 1])
                if close[i] > supertrend[i]:
                    supertrend[i] = lower_band
                    direction[i] = 1
                else:
                    direction[i] = -1
    
    return supertrend, direction


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # EMA calculation
    def ema(arr, period):
        result = np.zeros(len(arr))
        multiplier = 2 / (period + 1)
        result[period - 1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (arr[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = ema(macd_line, signal)
    
    # Histogram
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
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # First period
    sum_plus_dm = np.sum(plus_dm[1:period + 1])
    sum_minus_dm = np.sum(minus_dm[1:period + 1])
    sum_tr = np.sum(tr[1:period + 1])
    
    if sum_tr > 0:
        plus_di[period] = 100 * sum_plus_dm / sum_tr
        minus_di[period] = 100 * sum_minus_dm / sum_tr
    
    if plus_di[period] + minus_di[period] > 0:
        dx[period] = 100 * abs(plus_di[period] - minus_di[period]) / (plus_di[period] + minus_di[period])
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    # Wilder's smoothing for subsequent periods
    for i in range(period + 1, n):
        sum_plus_dm = plus_dm[i] + (sum_plus_dm - plus_dm[i - period])
        sum_minus_dm = minus_dm[i] + (sum_minus_dm - minus_dm[i - period])
        sum_tr = tr[i] + (sum_tr - tr[i - period])
        
        if sum_tr > 0:
            plus_di[i] = 100 * sum_plus_dm / sum_tr
            minus_di[i] = 100 * sum_minus_dm / sum_tr
        
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        if i >= period * 2:
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    adx_15m = calculate_adx(high, low, close, period=14)
    
    # 4h Supertrend and KAMA for trend (resample 15m → 4h)
    df_15m = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_15m.index = pd.date_range(start='2021-01-01', periods=n, freq='15min')
    
    # Resample to 4h
    df_4h = df_15m.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # 4h Supertrend for trend direction
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # 4h KAMA for trend confirmation
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    
    # Map 4h indicators back to 15m timeframe (16 x 15m = 4h)
    trend_15m = np.zeros(n)
    kama_confirm_15m = np.zeros(n)
    
    idx_15m_to_4h = np.arange(n) // 16
    
    for i in range(n):
        idx_4h = idx_15m_to_4h[i]
        if idx_4h < len(st_direction_4h) and idx_4h >= 10:
            # Supertrend direction
            trend_15m[i] = st_direction_4h[idx_4h]
            
            # KAMA confirmation: price above KAMA = bullish, below = bearish
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                kama_confirm_15m[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                kama_confirm_15m[i] = -1
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.35
    SIZE_HALF = 0.20
    
    # ADX threshold for trend strength
    ADX_MIN = 25
    
    # MACD histogram thresholds for entry
    MACD_ENTRY_THRESH = 0  # Cross above/below zero
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02
    
    first_valid = max(100, 14 * 2, 26 + 9)  # Wait for all indicators
    
    # Track position state
    entry_price = np.zeros(n)
    position_side = np.zeros(n)
    tp_triggered = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(macd_hist[i]) or np.isnan(adx_15m[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_15m[i]
        kama_conf = kama_confirm_15m[i]
        adx_val = adx_15m[i]
        macd_hist_val = macd_hist[i]
        prev_macd_hist = macd_hist[i - 1] if i > 0 else 0
        atr = atr_15m[i]
        price = close[i]
        
        # ATR filter - avoid extreme volatility
        if atr > 0 and atr / price > 0.05:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # MACD histogram exit (momentum fading)
                if prev_macd_hist > 0 and macd_hist_val < 0:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # MACD histogram exit (momentum fading)
                if prev_macd_hist < 0 and macd_hist_val > 0:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            continue
        
        # ADX filter - only trade strong trends
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        position_size = SIZE_FULL * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_HALF, position_size))
        
        # Entry logic: trend + KAMA confirmation + MACD histogram cross
        if trend == 1 and kama_conf == 1:  # Bullish trend confirmed
            # MACD histogram crosses above zero (momentum turning positive)
            if prev_macd_hist <= 0 and macd_hist_val > 0:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                
        elif trend == -1 and kama_conf == -1:  # Bearish trend confirmed
            # MACD histogram crosses below zero (momentum turning negative)
            if prev_macd_hist >= 0 and macd_hist_val < 0:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals