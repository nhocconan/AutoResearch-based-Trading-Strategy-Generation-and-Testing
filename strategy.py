#!/usr/bin/env python3
"""
Experiment #1515: 6h Primary + 12h/1d HTF — Fisher Transform + Vol Spike Mean Reversion

Hypothesis: 6h timeframe captures multi-day swings better than 4h (less noise) and 12h 
(more opportunities). This strategy combines:
1. 12h HMA(21) for intermediate trend direction
2. 1d HMA(21) for major bias filter (avoid counter-trend in strong moves)
3. Ehlers Fisher Transform(9) for precise reversal entry timing
4. Choppiness Index(14) for regime detection (trend vs range)
5. ATR ratio(7/30) for vol spike detection (panic capitulation entries)

Why this should work on 6h:
- Fisher Transform excels at catching reversals in bear market rallies (2025 test period)
- Vol spike reversion captures "panic bottom" entries after ATR expansion
- 12h HMA provides smoother trend signal than 4h, faster than 1d
- 6h natural frequency: ~40-50 trades/year (fee-efficient)
- LOOSE Fisher thresholds (-1.5/+1.5) guarantee trade generation

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG trend: 12h_HMA bullish + 1d_HMA bullish + Fisher < -1.0 + CHOP < 50
- SHORT trend: 12h_HMA bearish + 1d_HMA bearish + Fisher > +1.0 + CHOP < 50
- LONG vol-spike: ATR_ratio > 1.8 + Fisher < -1.2 + price < BB_lower (panic capitulation)
- SHORT vol-spike: ATR_ratio > 1.8 + Fisher > +1.2 + price > BB_upper (euphoria top)

Target: Sharpe>0.7, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_vol_spike_regime_12h1d_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points with sharp peaks
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close := (high + low) / 2)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate median price
    median = (high + low) / 2
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        window = median[i - period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        
        highest = np.max(window)
        lowest = np.min(window)
        price_range = highest - lowest
        
        if price_range < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        normalized = (median[i] - lowest) / price_range
        
        # Clamp to avoid division issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with previous value (Ehlers method)
        if i > period - 1 and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for vol spike detection"""
    n = len(close)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_short = pd.Series(tr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(tr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    ratio = np.full(n, np.nan, dtype=np.float64)
    mask = atr_long > 1e-10
    ratio[mask] = atr_short[mask] / atr_long[mask]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(bb_lower[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 50.0  # Looser threshold for more trades
        is_range_regime = chop > 55.0
        
        # === TREND DIRECTION (12h HMA + 1d HMA bias) ===
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 12h and 1d aligned (stronger signal)
        trend_bullish = price_above_12h and price_above_1d
        trend_bearish = price_below_12h and price_below_1d
        
        # === FISHER TRANSFORM ===
        fisher_val = fisher[i]
        fisher_prev_val = fisher_prev[i]
        
        # Fisher crossovers for entry timing
        fisher_cross_up = fisher_prev_val < -1.0 and fisher_val >= -1.0
        fisher_cross_down = fisher_prev_val > 1.0 and fisher_val <= 1.0
        
        fisher_oversold = fisher_val < -1.2
        fisher_overbought = fisher_val > 1.2
        
        # === VOL SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 1.8  # ATR(7) > 1.8x ATR(30)
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.005  # within 0.5% of lower band
        bb_touch_upper = close[i] >= bb_upper[i] * 0.995  # within 0.5% of upper band
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Fisher reversal with trend bias
        if is_trend_regime:
            # LONG: bullish trend + Fisher oversold or cross up
            if trend_bullish and (fisher_oversold or fisher_cross_up):
                desired_signal = SIZE_BASE
            
            # SHORT: bearish trend + Fisher overbought or cross down
            elif trend_bearish and (fisher_overbought or fisher_cross_down):
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Fisher mean reversion
        elif is_range_regime:
            # LONG: Fisher deeply oversold
            if fisher_val < -1.5:
                desired_signal = SIZE_BASE
            
            # SHORT: Fisher deeply overbought
            elif fisher_val > 1.5:
                desired_signal = -SIZE_BASE
        
        # VOL SPIKE REVERSION (panic/euphoria capitulation)
        if vol_spike:
            # LONG: vol spike + Fisher oversold + price at BB lower (panic bottom)
            if fisher_oversold and bb_touch_lower:
                desired_signal = SIZE_STRONG
            
            # SHORT: vol spike + Fisher overbought + price at BB upper (euphoria top)
            elif fisher_overbought and bb_touch_upper:
                desired_signal = -SIZE_STRONG
        
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