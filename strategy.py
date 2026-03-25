#!/usr/bin/env python3
"""
Experiment #1158: 4h Primary + 1d HTF — ADX Regime + Dual Mode Strategy (Loose Entries)

Hypothesis: Using ADX to detect market regime (trending vs ranging) and switching
between trend-following and mean-reversion modes will capture both market types.
Key improvement over #1002: LOOSER entry conditions to guarantee trade generation.

Key components:
1. ADX(14) regime: >22 = trend, <25 = range (overlapping for stability)
2. 1d HMA(21) for long-term bias alignment
3. Trend mode: Enter when price aligns with 1d HMA + RSI confirmation
4. Range mode: Fade Bollinger Band extremes (1.5 std) with RSI extremes
5. ATR(14) 2.5x trailing stop
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to guarantee 30+ trades):
- LONG trend: ADX>20 + price>1d_HMA*0.98 + RSI(14)>40
- LONG range: price<BB_lower(1.5) + RSI(14)<40
- SHORT trend: ADX>20 + price<1d_HMA*1.02 + RSI(14)<60
- SHORT range: price>BB_upper(1.5) + RSI(14)>60

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_adx_regime_dual_mode_loose_1d_v2"
timeframe = "4h"
leverage = 1.0

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

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    smoothed_tr = np.zeros(n, dtype=np.float64)
    smoothed_pm = np.zeros(n, dtype=np.float64)
    smoothed_nm = np.zeros(n, dtype=np.float64)
    
    smoothed_tr[period-1] = np.sum(tr[:period])
    smoothed_pm[period-1] = np.sum(plus_dm[:period])
    smoothed_nm[period-1] = np.sum(minus_dm[:period])
    
    for i in range(period, n):
        smoothed_tr[i] = smoothed_tr[i-1] - smoothed_tr[i-1]/period + tr[i]
        smoothed_pm[i] = smoothed_pm[i-1] - smoothed_pm[i-1]/period + plus_dm[i]
        smoothed_nm[i] = smoothed_nm[i-1] - smoothed_nm[i-1]/period + minus_dm[i]
    
    di_plus = np.full(n, np.nan, dtype=np.float64)
    di_minus = np.full(n, np.nan, dtype=np.float64)
    dx = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period-1, n):
        if smoothed_tr[i] > 1e-10:
            di_plus[i] = 100.0 * smoothed_pm[i] / smoothed_tr[i]
            di_minus[i] = 100.0 * smoothed_nm[i] / smoothed_tr[i]
        
        if di_plus[i] + di_minus[i] > 1e-10:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    return adx

def calculate_bollinger(close, period=20, std_mult=1.5):
    """Bollinger Bands with configurable std multiplier"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=1.5)
    
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
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (ADX) ===
        # Overlapping thresholds for stability
        is_trending = adx_14[i] > 20.0
        is_ranging = adx_14[i] < 25.0
        
        # === HTF BIAS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i] * 0.995
        hma_1d_bear = close[i] < hma_1d_aligned[i] * 1.005
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # TREND MODE entries (ADX > 20)
        if is_trending:
            # Long in uptrend with RSI confirmation
            if hma_1d_bull and rsi_14[i] > 40.0 and rsi_14[i] < 75.0:
                desired_signal = SIZE_STRONG if rsi_14[i] < 55.0 else SIZE_BASE
            # Short in downtrend with RSI confirmation
            elif hma_1d_bear and rsi_14[i] < 60.0 and rsi_14[i] > 25.0:
                desired_signal = -SIZE_STRONG if rsi_14[i] > 45.0 else -SIZE_BASE
        
        # RANGE MODE entries (ADX < 25)
        if is_ranging:
            # Long at lower Bollinger with oversold RSI
            if close[i] < bb_lower[i] and rsi_14[i] < 40.0:
                desired_signal = SIZE_STRONG if rsi_14[i] < 30.0 else SIZE_BASE
            # Short at upper Bollinger with overbought RSI
            elif close[i] > bb_upper[i] and rsi_14[i] > 60.0:
                desired_signal = -SIZE_STRONG if rsi_14[i] > 70.0 else -SIZE_BASE
        
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