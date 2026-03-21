#!/usr/bin/env python3
"""
EXPERIMENT #008 - KAMA Adaptive Trend + MACD Histogram + ADX Strength Filter
====================================================================================
Hypothesis: Use 4h KAMA (Kaufman Adaptive Moving Average) for trend direction - it adapts
to volatility regimes better than EMA (less whipsaw in chop, faster in trends). Use 1h
MACD histogram crosses for entry timing (momentum confirmation). Add ADX(14) strength
filter to only trade when trend strength > 25 (avoid weak/choppy markets). Keep proven
position sizing (0.35 max, discrete levels) and ATR stoploss (2*ATR).

Why this might beat Sharpe=5.525:
- KAMA adapts to market regime automatically (ER-based smoothing)
- MACD histogram cross provides momentum confirmation (different from RSI pullback)
- ADX filter avoids trading in weak trend conditions (reduces false signals)
- Multi-timeframe: 4h trend + 1h entries (proven architecture from #005/#007)
- Discrete signal levels (0.0, ±0.20, ±0.35) minimize churn costs
- ATR-based stoploss protects against adverse moves
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_adx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market efficiency ratio (ER)
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    Higher ER = trending market = faster smoothing
    Lower ER = choppy market = slower smoothing
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    # SC = ER * (fast_SC - slow_SC) + slow_SC
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """
    Calculate MACD line, signal line, and histogram
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(signal_period) of MACD
    Histogram = MACD - Signal
    """
    n = len(close)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    if n < slow + signal_period:
        return macd_line, signal_line, histogram
    
    # Calculate EMAs
    def ema(data, period):
        result = np.zeros(len(data))
        multiplier = 2.0 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Calculate signal line (EMA of MACD)
    valid_macd = macd_line[slow - 1:]
    signal_values = ema(valid_macd, signal_period)
    
    for i in range(len(signal_values)):
        signal_line[slow - 1 + i] = signal_values[i]
    
    # Calculate histogram
    for i in range(slow - 1 + signal_period - 1, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = weak/choppy
    """
    n = len(close)
    adx = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    # Calculate True Range and DM
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
    
    # Smooth TR, +DM, -DM using Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    # Initial averages
    if n >= 2 * period:
        avg_tr = np.mean(tr[1:period + 1])
        avg_plus_dm = np.mean(plus_dm[1:period + 1])
        avg_minus_dm = np.mean(minus_dm[1:period + 1])
        
        for i in range(period, n):
            if i == period:
                pass  # Use initial values
            else:
                avg_tr = (avg_tr * (period - 1) + tr[i]) / period
                avg_plus_dm = (avg_plus_dm * (period - 1) + plus_dm[i]) / period
                avg_minus_dm = (avg_minus_dm * (period - 1) + minus_dm[i]) / period
            
            if avg_tr > 0:
                plus_di[i] = 100 * avg_plus_dm / avg_tr
                minus_di[i] = 100 * avg_minus_dm / avg_tr
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        
        # Calculate ADX (smoothed DX)
        adx[2 * period - 1] = np.mean(dx[period:2 * period])
        for i in range(2 * period, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


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
    macd_1h, signal_1h, hist_1h = calculate_macd(close, fast=12, slow=26, signal_period=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(high, low, close, period=14)
    
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
    
    # Calculate 4h KAMA trend
    kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    
    # Calculate 4h ADX for trend strength
    adx_4h, _, _ = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h indicators back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    adx_1h_from_4h = np.zeros(n)
    kama_1h_from_4h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(kama_4h):
            kama_1h_from_4h[i] = kama_4h[idx_4h]
            adx_1h_from_4h[i] = adx_4h[idx_4h]
            # Trend direction: price above KAMA = bullish, below = bearish
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                trend_1h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                trend_1h[i] = -1
            else:
                trend_1h[i] = 0
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    
    # MACD histogram thresholds for entry
    MACD_LONG_THRESHOLD = 0.0    # Histogram crosses above 0
    MACD_SHORT_THRESHOLD = 0.0   # Histogram crosses below 0
    
    # ADX strength filter
    ADX_MIN = 25.0    # Only trade when ADX > 25 (strong trend)
    
    # Z-score filter thresholds
    ZSCORE_MAX = 2.5    # Don't enter if price is > 2.5 std dev from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(60, 30, 14, 28, 50)
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    
    # Track MACD histogram for cross detection
    prev_hist = np.zeros(n)
    
    for i in range(first_valid, n):
        prev_hist[i] = hist_1h[i - 1] if i > 0 else 0
        
        if np.isnan(hist_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(adx_1h_from_4h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        hist_val = hist_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        adx_4h_val = adx_1h_from_4h[i]
        kama_val = kama_1h_from_4h[i]
        
        # Z-score filter - don't enter at extreme valuations
        if abs(zscore_val) > ZSCORE_MAX:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ADX strength filter - only trade when trend is strong
        if adx_4h_val < ADX_MIN:
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
            
            if prev_side == 1:
                # Stoploss check (2*ATR against position)
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
                
                # MACD histogram reversal exit (momentum fading)
                if hist_val < MACD_LONG_THRESHOLD and prev_hist[i] >= MACD_LONG_THRESHOLD:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
            elif prev_side == -1:
                # Stoploss check (2*ATR against position)
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
                
                # MACD histogram reversal exit (momentum fading)
                if hist_val > MACD_SHORT_THRESHOLD and prev_hist[i] <= MACD_SHORT_THRESHOLD:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
        
        # Generate new entries based on trend + MACD histogram cross
        if trend == 1:  # 4h uptrend (price > KAMA)
            # MACD histogram crosses above 0 (bullish momentum)
            if hist_val > MACD_LONG_THRESHOLD and prev_hist[i] <= MACD_LONG_THRESHOLD:
                if position_side[i - 1] != -1:
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
            else:
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    
        elif trend == -1:  # 4h downtrend (price < KAMA)
            # MACD histogram crosses below 0 (bearish momentum)
            if hist_val < MACD_SHORT_THRESHOLD and prev_hist[i] >= MACD_SHORT_THRESHOLD:
                if position_side[i - 1] != 1:
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
            else:
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
    
    return signals