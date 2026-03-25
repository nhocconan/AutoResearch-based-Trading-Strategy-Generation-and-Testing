#!/usr/bin/env python3
"""
Experiment #1295: 6h Primary + 12h/1d HTF — Regime-Adaptive Mean Reversion Within Trend

Hypothesis: Pure trend following fails on 6h because entries occur at exhaustion points.
Pure mean reversion fails because it fights the major trend. This strategy combines both:

1. 1d HMA(21) for MAJOR trend bias (only trade WITH daily trend direction)
2. 12h ATR ratio (ATR7/ATR30) for VOLATILITY regime (high vol=trend, low vol=mean-revert)
3. 6h RSI(7) for ENTRY timing (oversold in uptrend, overbought in downtrend)
4. 6h Donchian(20) breakout confirmation (avoids entering during consolidation)

Why this should work:
- 1d trend filter = never fight the major trend (solves mean-reversion failure)
- RSI pullback = enter at better prices within trend (solves trend exhaustion)
- ATR ratio regime = adapt entry style to volatility (high vol needs breakout confirm)
- Loose RSI thresholds (25/75 instead of 20/80) = guarantee 30-60 trades/year
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn

Entry logic:
- LONG: 1d_HMA bullish + RSI(7)<35 + (ATR_ratio>1.5 → Donchian break OR ATR_ratio<1.5 → direct)
- SHORT: 1d_HMA bearish + RSI(7)>65 + (ATR_ratio>1.5 → Donchian break OR ATR_ratio<1.5 → direct)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_adaptive_rsi_donchian_12h1d_v1"
timeframe = "6h"
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
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 12h ATR ratio for volatility regime
    atr_12h_7_raw = calculate_atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=7)
    atr_12h_30_raw = calculate_atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=30)
    
    atr_ratio_12h_raw = np.full(len(atr_12h_7_raw), np.nan, dtype=np.float64)
    for i in range(len(atr_12h_7_raw)):
        if not np.isnan(atr_12h_7_raw[i]) and not np.isnan(atr_12h_30_raw[i]) and atr_12h_30_raw[i] > 1e-10:
            atr_ratio_12h_raw[i] = atr_12h_7_raw[i] / atr_12h_30_raw[i]
    
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h_raw)
    
    # Calculate 6h indicators
    atr_6h_14 = calculate_atr(high, low, close, period=14)
    rsi_6h_7 = calculate_rsi(close, period=7)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_6h_14[i]) or atr_6h_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_6h_7[i]):
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MAJOR TREND BIAS (1d HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME (12h ATR ratio) ===
        atr_ratio = atr_ratio_12h_aligned[i]
        high_vol_regime = (not np.isnan(atr_ratio)) and (atr_ratio > 1.5)
        low_vol_regime = (not np.isnan(atr_ratio)) and (atr_ratio <= 1.5)
        
        # === RSI PULLBACK ===
        rsi = rsi_6h_7[i]
        rsi_oversold = rsi < 35  # Loose threshold for more trades
        rsi_overbought = rsi > 65  # Loose threshold for more trades
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + RSI oversold + (high vol needs breakout OR low vol direct entry)
        if price_above_1d and rsi_oversold:
            if high_vol_regime and donchian_breakout_long:
                desired_signal = SIZE_STRONG  # High vol breakout = strong signal
            elif low_vol_regime:
                desired_signal = SIZE_BASE  # Low vol mean reversion = base signal
        
        # SHORT: 1d bearish + RSI overbought + (high vol needs breakout OR low vol direct entry)
        elif price_below_1d and rsi_overbought:
            if high_vol_regime and donchian_breakout_short:
                desired_signal = -SIZE_STRONG  # High vol breakout = strong signal
            elif low_vol_regime:
                desired_signal = -SIZE_BASE  # Low vol mean reversion = base signal
        
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
                entry_atr = atr_6h_14[i]
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