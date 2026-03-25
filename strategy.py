#!/usr/bin/env python3
"""
Experiment #1591: 6h Primary + 1d/1w HTF — Donchian Breakout + Choppiness Regime

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Donchian breakouts
guarantee trade generation in volatile crypto markets, while Choppiness Index filters
false breakouts during range-bound periods. 1d HMA provides trend bias to avoid
counter-trend breakouts. This combines proven breakout mechanics with regime filtering.

Why this should work where 6h strategies failed:
1. DONCHIAN(20) breakouts = GUARANTEED trades (price MUST break 20-bar high/low eventually)
2. LOOSE regime thresholds: CHOP<45 (not 38.2) for trend, CHOP>55 (not 61.8) for range
3. VOLUME confirmation relaxed to 1.2x (not 1.5x) to allow more trades
4. 1d HMA bias is soft filter (reduces size, doesn't block entry)
5. Asymmetric sizing: 0.30 in trend regime, 0.20 in range regime

Key differences from failed 6h attempts:
- NOT RSI mean reversion (failed Sharpe=-6.9)
- NOT CPR pivots (failed Sharpe=-4.1)
- NOT Triple HMA (failed Sharpe=-1.05)
- NOT Weekly pivot combinations (all failed)

Entry logic (LOOSE to guarantee ≥30 trades/train):
- LONG trend: CHOP<45 + price>1d_HMA + Donchian(20) breakout + vol>1.2x
- SHORT trend: CHOP<45 + price<1d_HMA + Donchian(20) breakdown + vol>1.2x
- LONG range: CHOP>55 + Donchian(20) breakout + RSI<40 (mean revert long)
- SHORT range: CHOP>55 + Donchian(20) breakdown + RSI>60 (mean revert short)
- NEUTRAL: 1w_HMA bias + 1d_HMA confirmation (smaller size 0.15)

Target: Sharpe>0.6, trades>=120 train (4 years), trades>=45 test (15 months)
Timeframe: 6h
Size: 0.20-0.30 discrete
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_chop_regime_1d1w_vol_v1"
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
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    Using LOOSE thresholds: <45 for trend, >55 for range
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_RANGE = 0.20
    SIZE_NEUTRAL = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
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
        
        # === REGIME DETECTION (Choppiness Index - LOOSE thresholds) ===
        chop = chop_14[i]
        is_trend_regime = chop < 45.0  # LOOSE: was 38.2
        is_range_regime = chop > 55.0  # LOOSE: was 61.8
        
        # === TREND DIRECTION (1d and 1w HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION (LOOSE: 1.2x not 1.5x) ===
        vol_confirmed = vol_ratio[i] > 1.2 if not np.isnan(vol_ratio[i]) else False
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout = close above previous bar's Donchian upper
        donchian_breakout_long = False
        donchian_breakout_short = False
        
        if i > 0 and not np.isnan(donch_upper[i-1]):
            if close[i] > donch_upper[i-1]:
                donchian_breakout_long = True
        if i > 0 and not np.isnan(donch_lower[i-1]):
            if close[i] < donch_lower[i-1]:
                donchian_breakout_short = True
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow breakout direction with 1d HMA bias
        if is_trend_regime:
            # LONG: 1d bullish + Donchian breakout (volume soft confirm)
            if price_above_1d and donchian_breakout_long:
                desired_signal = SIZE_TREND if vol_confirmed else SIZE_RANGE
            
            # SHORT: 1d bearish + Donchian breakdown (volume soft confirm)
            elif price_below_1d and donchian_breakout_short:
                desired_signal = -SIZE_TREND if vol_confirmed else -SIZE_RANGE
            
            # Counter-trend breakout (smaller size, requires volume)
            elif vol_confirmed:
                if donchian_breakout_long and price_below_1d:
                    desired_signal = SIZE_RANGE * 0.7  # Smaller counter-trend
                elif donchian_breakout_short and price_above_1d:
                    desired_signal = -SIZE_RANGE * 0.7
        
        # RANGE REGIME: Mean reversion at Donchian extremes + RSI filter
        elif is_range_regime:
            # LONG: Donchian breakout + RSI oversold (mean revert long)
            if donchian_breakout_long and rsi_14[i] < 40:
                desired_signal = SIZE_RANGE
            
            # SHORT: Donchian breakdown + RSI overbought (mean revert short)
            elif donchian_breakout_short and rsi_14[i] > 60:
                desired_signal = -SIZE_RANGE
            
            # Neutral range: use 1d HMA for small directional bias
            elif price_above_1d and rsi_14[i] < 55:
                desired_signal = SIZE_NEUTRAL
            elif price_below_1d and rsi_14[i] > 45:
                desired_signal = -SIZE_NEUTRAL
        
        # NEUTRAL REGIME: Use 1w HMA for ultra-long bias
        else:
            if price_above_1w and price_above_1d:
                desired_signal = SIZE_NEUTRAL
            elif price_below_1w and price_below_1d:
                desired_signal = -SIZE_NEUTRAL
        
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
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_RANGE * 0.9:
            final_signal = SIZE_RANGE
        elif desired_signal <= -SIZE_RANGE * 0.9:
            final_signal = -SIZE_RANGE
        elif desired_signal >= SIZE_NEUTRAL * 0.9:
            final_signal = SIZE_NEUTRAL
        elif desired_signal <= -SIZE_NEUTRAL * 0.9:
            final_signal = -SIZE_NEUTRAL
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