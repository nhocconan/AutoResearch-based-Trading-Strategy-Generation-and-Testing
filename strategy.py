#!/usr/bin/env python3
"""
Experiment #1280: 6h Primary + 1d/1w HTF — Donchian Breakout with Weekly Regime Bias

Hypothesis: The current best 6h strategy (KAMA+ROC) achieved Sharpe=0.447 but may miss
large breakout moves. This variant uses Donchian channel breakouts which capture
major trend moves more effectively, especially in crypto's volatile regime.

Key innovations:
1. 1w HMA(21) for major regime bias (ONLY direction filter, not entry filter)
2. 1d HMA(21) for intermediate trend confirmation
3. 6h Donchian(20) breakout for entries (price breaks 20-bar high/low)
4. 6h ATR(14) expansion filter (breakout must have vol support)
5. 2.5x ATR trailing stoploss for risk management
6. LOOSE entry conditions to guarantee 30-60 trades/year

Why this should work on 6h:
- 6h Donchian(20) = ~5 day breakout window (catches major moves without noise)
- Weekly bias prevents counter-trend trades in strong regimes
- ATR expansion filter avoids false breakouts in low vol
- Fewer conditions than failed strategies = more trades guaranteed
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn

Entry logic (LOOSE):
- LONG: 1w_HMA bullish + 1d_HMA rising + Donchian(20) high break + ATR expanding
- SHORT: 1w_HMA bearish + 1d_HMA falling + Donchian(20) low break + ATR expanding

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_breakout_weekly_bias_1d1w_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """Ehlers Fisher Transform - normalizes price to -1 to +1 range"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        hl2 = (high[i] + low[i]) / 2.0
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        value = (hl2 - lowest) / (highest - lowest)
        value = max(0.001, min(0.999, value))  # clamp to avoid log(0)
        
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        if i >= 1 and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]  # smooth
    
    return fisher

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
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    fisher_9 = calculate_fisher_transform(high, low, period=9)
    
    # ATR ratio for volatility expansion
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            atr_ratio[i] = atr_14[i] / atr_30[i]
    
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
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
        
        # === WEEKLY REGIME BIAS (loose filter - just direction) ===
        # 1w HMA slope (compare to 2 bars ago for stability)
        hma_1w_slope = 0.0
        if i >= 2 and not np.isnan(hma_1w_aligned[i-2]):
            hma_1w_slope = hma_1w_aligned[i] - hma_1w_aligned[i-2]
        
        weekly_bullish = hma_1w_slope >= 0
        weekly_bearish = hma_1w_slope < 0
        
        # === DAILY TREND CONFIRMATION ===
        # 1d HMA slope
        hma_1d_slope = 0.0
        if i >= 2 and not np.isnan(hma_1d_aligned[i-2]):
            hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-2]
        
        daily_bullish = hma_1d_slope > 0
        daily_bearish = hma_1d_slope < 0
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Check if price broke above upper or below lower
        breakout_long = close[i] > donchian_upper[i-1]  # break previous bar's upper
        breakout_short = close[i] < donchian_lower[i-1]  # break previous bar's lower
        
        # === VOLATILITY EXPANSION FILTER ===
        # ATR must be expanding (ratio > 1.0 means short-term vol > long-term)
        vol_expanding = False
        if not np.isnan(atr_ratio[i]) and atr_ratio[i] > 0.9:
            vol_expanding = True
        
        # === FISHER TRANSFORM CONFIRMATION (optional boost) ===
        fisher_confirmed_long = False
        fisher_confirmed_short = False
        if not np.isnan(fisher_9[i]):
            # Fisher crossing up from oversold
            if i >= 1 and not np.isnan(fisher_9[i-1]):
                if fisher_9[i] > -1.0 and fisher_9[i-1] <= -1.0:
                    fisher_confirmed_long = True
                elif fisher_9[i] < 1.0 and fisher_9[i-1] >= 1.0:
                    fisher_confirmed_short = True
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: Weekly bias OK + Daily rising + Donchian breakout + Vol expansion
        if weekly_bullish and daily_bullish and breakout_long:
            if vol_expanding:
                if fisher_confirmed_long:
                    desired_signal = SIZE_STRONG  # All signals align
                else:
                    desired_signal = SIZE_BASE  # Basic breakout
        
        # SHORT: Weekly bias OK + Daily falling + Donchian breakout + Vol expansion
        elif weekly_bearish and daily_bearish and breakout_short:
            if vol_expanding:
                if fisher_confirmed_short:
                    desired_signal = -SIZE_STRONG  # All signals align
                else:
                    desired_signal = -SIZE_BASE  # Basic breakout
        
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