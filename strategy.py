#!/usr/bin/env python3
"""
Experiment #1031: 6h Primary + 1d/1w HTF — SuperTrend + ADX + Triple HMA Filter

Hypothesis: SuperTrend (ATR-based trend following) provides clear directional signals with
built-in stop levels. Combined with 1d/1w HMA for trend bias and ADX for trend strength
filtering, this should capture multi-day swings while avoiding choppy whipsaws.

Key innovations:
1. SuperTrend(10, 3.0): ATR-based trend indicator with clear long/short signals
   - More robust than EMA crossover in volatile crypto markets
   - Built-in trailing stop mechanism
2. ADX(14) > 20 filter: Only trade when trend strength is sufficient
   - Avoids mean-reversion chop where SuperTrend whipsaws
3. Triple HMA alignment: 6h price vs 1d HMA vs 1w HMA
   - Long only when 6h > 1d_HMA > 1w_HMA (bullish stack)
   - Short only when 6h < 1d_HMA < 1w_HMA (bearish stack)
4. RSI(14) momentum filter: RSI > 45 for longs, RSI < 55 for shorts
   - Ensures momentum confirms the trend direction
5. ATR(14) 2.5x trailing stop: Additional risk management beyond SuperTrend
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work on 6h:
- 6h captures 2-4 day swings (optimal for crypto trend following)
- SuperTrend adapts to volatility (wider stops in high vol, tighter in low vol)
- ADX filter avoids the #1 killer: trend-following in ranging markets (2022-2023)
- Triple HMA ensures we only trade with multi-TF alignment
- Target: 30-50 trades/year (strict entry filters)

Entry conditions:
- LONG: SuperTrend=long + ADX>20 + price>1d_HMA>1w_HMA + RSI>45
- SHORT: SuperTrend=short + ADX>20 + price<1d_HMA<1w_HMA + RSI<55

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_supertrend_adx_triple_hma_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range - Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n, dtype=np.float64)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    SuperTrend Indicator - ATR-based trend following
    Returns: (supertrend_values, trend_direction)
    trend_direction: +1 for long, -1 for short
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    
    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)
    supertrend = np.full(n, np.nan, dtype=np.float64)
    trend = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            # Upper band logic
            if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = upper_band[i-1]
            
            # Lower band logic
            if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = lower_band[i-1]
            
            # Trend determination
            if trend[i-1] == 1:
                if close[i] < lower_band[i]:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
            else:
                if close[i] > upper_band[i]:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
    
    return supertrend, trend

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = strong trend, ADX < 20 = weak/ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Wilder's smoothing
    atr = np.zeros(n, dtype=np.float64)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    atr[period-1] = np.mean(tr[:period])
    plus_di[period-1] = 100.0 * np.mean(plus_dm[:period]) / atr[period-1] if atr[period-1] > 0 else 0
    minus_di[period-1] = 100.0 * np.mean(minus_dm[:period]) / atr[period-1] if atr[period-1] > 0 else 0
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        prev_plus = (plus_di[i-1] * atr[i-1] / 100.0) if atr[i-1] > 0 else 0
        prev_minus = (minus_di[i-1] * atr[i-1] / 100.0) if atr[i-1] > 0 else 0
        
        smooth_plus = prev_plus * (period - 1) + plus_dm[i]
        smooth_minus = prev_minus * (period - 1) + minus_dm[i]
        
        plus_di[i] = 100.0 * smooth_plus / atr[i] if atr[i] > 0 else 0
        minus_di[i] = 100.0 * smooth_minus / atr[i] if atr[i] > 0 else 0
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = SMA of DX
    adx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period * 2 - 1, n):
        adx[i] = np.mean(dx[i-period+1:i+1])
    
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    
    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])
    
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    supertrend_vals, supertrend_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(supertrend_direction[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (Triple HMA alignment) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong trend alignment (all TFs agree)
        strong_bull = hma_1d_bull and hma_1w_bull and hma_1d_aligned[i] > hma_1w_aligned[i]
        strong_bear = hma_1d_bear and hma_1w_bear and hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        is_trending = adx_14[i] > 20.0  # Minimum trend strength
        
        # === SUPERTREND DIRECTION ===
        st_long = supertrend_direction[i] == 1
        st_short = supertrend_direction[i] == -1
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: SuperTrend long + trending + HTF bullish + RSI momentum
        if st_long and is_trending and strong_bull and rsi_14[i] > 45.0 and rsi_14[i] < 80.0:
            desired_signal = SIZE_STRONG
        elif st_long and is_trending and hma_1d_bull and hma_1w_bull and rsi_14[i] > 50.0:
            desired_signal = SIZE_BASE
        
        # SHORT: SuperTrend short + trending + HTF bearish + RSI momentum
        elif st_short and is_trending and strong_bear and rsi_14[i] < 55.0 and rsi_14[i] > 20.0:
            desired_signal = -SIZE_STRONG
        elif st_short and is_trending and hma_1d_bear and hma_1w_bear and rsi_14[i] < 50.0:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals