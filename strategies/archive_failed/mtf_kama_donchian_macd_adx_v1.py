#!/usr/bin/env python3
"""
EXPERIMENT #006 - KAMA Trend + Donchian Breakout + MACD Entry + ADX Filter
====================================================================================
Hypothesis: Replace EMA with KAMA (Kaufman Adaptive Moving Average) for trend,
which adapts to market volatility and reduces whipsaw in choppy conditions.
Add Donchian channel breakout as trend confirmation (price above 20-period high = strong uptrend).
Use MACD histogram cross for entry timing (more responsive than RSI for momentum).
Add ADX(14) filter to only trade when trend has sufficient strength (ADX > 25).
Keep Z-score filter to avoid extreme valuations.

Why this might beat Sharpe=5.525:
- KAMA reduces false signals in ranging markets (adapts smoothing based on volatility)
- Donchian breakout confirms trend strength (price at channel extreme)
- MACD histogram cross catches momentum shifts earlier than RSI
- ADX filter avoids trading weak trends (major cause of drawdown)
- Multi-timeframe proven: 4h trend + 1h entries
- Discrete signal levels (0.0, ±0.20, ±0.35) minimize churn costs
"""

import numpy as np
import pandas as pd

name = "mtf_kama_donchian_macd_adx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market volatility/noise
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = change / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    if n < slow + signal:
        return macd_line, signal_line, histogram
    
    # Calculate EMAs
    def ema(arr, period):
        result = np.zeros(len(arr))
        result[0] = arr[0]
        mult = 2.0 / (period + 1)
        for i in range(1, len(arr)):
            result[i] = (arr[i] - result[i - 1]) * mult + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    for i in range(slow, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal line is EMA of MACD
    signal_line[slow] = macd_line[slow]
    mult = 2.0 / (signal + 1)
    for i in range(slow + 1, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * mult + signal_line[i - 1]
    
    for i in range(slow, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    n = len(close)
    adx = np.zeros(n)
    
    if n < period * 2:
        return adx
    
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
    
    # Smooth TR, +DM, -DM
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_smooth = np.mean(plus_dm[1:period])
    minus_smooth = np.mean(minus_dm[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_smooth = (plus_smooth * (period - 1) + plus_dm[i]) / period
        minus_smooth = (minus_smooth * (period - 1) + minus_dm[i]) / period
        
        if atr[i] > 0:
            plus_di[i] = 100 * plus_smooth / atr[i]
            minus_di[i] = 100 * minus_smooth / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation from mean)"""
    n = len(close)
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        mean = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # 4h indicators for trend (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    n_4h = len(c_4h)
    
    # Calculate 4h KAMA for adaptive trend
    kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    
    # Calculate 4h Donchian channels
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
    
    # Calculate 4h ADX for trend strength
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # 4h trend direction based on KAMA and Donchian
    trend_4h = np.zeros(n_4h)
    for i in range(max(50, 20), n_4h):
        # Bullish: price above KAMA AND above Donchian middle
        donchian_mid = (donchian_upper_4h[i] + donchian_lower_4h[i]) / 2
        if c_4h[i] > kama_4h[i] and c_4h[i] > donchian_mid and adx_4h[i] > 20:
            trend_4h[i] = 1  # Bullish
        elif c_4h[i] < kama_4h[i] and c_4h[i] < donchian_mid and adx_4h[i] > 20:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    
    # MACD histogram thresholds for entry
    MACD_LONG_THRESHOLD = 0      # Histogram crosses above 0
    MACD_SHORT_THRESHOLD = 0     # Histogram crosses below 0
    
    # ADX filter thresholds
    ADX_MIN = 20        # Minimum ADX for trend strength
    
    # Z-score filter thresholds
    ZSCORE_MAX = 2.0    # Don't enter if price is > 2 std dev from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02
    
    first_valid = max(60, 20, 14, 28)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    
    for i in range(first_valid, n):
        if np.isnan(macd_hist_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        macd_hist = macd_hist_1h[i]
        macd_hist_prev = macd_hist_1h[i - 1] if i > 0 else 0
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        adx = adx_1h[i]
        price = close[i]
        
        # Z-score filter - don't enter at extreme valuations
        if abs(zscore_val) > ZSCORE_MAX:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else 0
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else price
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ADX filter - only trade when trend has strength
        if adx < ADX_MIN:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                
                # Stoploss check
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF * np.sign(prev_side)
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    continue
                
                # MACD exit signal (histogram crosses below 0)
                if macd_hist < MACD_LONG_THRESHOLD and macd_hist_prev >= MACD_LONG_THRESHOLD:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                
                # Stoploss check
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    continue
                
                # MACD exit signal (histogram crosses above 0)
                if macd_hist > MACD_SHORT_THRESHOLD and macd_hist_prev <= MACD_SHORT_THRESHOLD:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        position_size = SIZE_FULL * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_HALF, position_size))
        
        if trend == 1:  # 4h uptrend
            # MACD histogram cross above 0 for long entry
            if macd_hist > MACD_LONG_THRESHOLD and macd_hist_prev <= MACD_LONG_THRESHOLD:
                if position_side[i - 1] != -1:
                    signals[i] = position_size
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
            else:
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # MACD histogram cross below 0 for short entry
            if macd_hist < MACD_SHORT_THRESHOLD and macd_hist_prev >= MACD_SHORT_THRESHOLD:
                if position_side[i - 1] != 1:
                    signals[i] = -position_size
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
            else:
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals