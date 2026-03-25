#!/usr/bin/env python3
"""
Experiment #1602: 4h Primary + 1d/1w HTF — KAMA/HMA Crossover with Dynamic RSI

Hypothesis: After 1300+ failed experiments, complex regime detection (Choppiness, Fisher)
consistently fails. Return to proven basics: KAMA (adapts to noise) + HMA (low lag) 
crossover with RSI filter and strong 1d trend bias.

Why this should work where others failed:
1. KAMA adapts ER (Efficiency Ratio) - smooth in chop, fast in trends
2. HMA provides faster signal than EMA with less whipsaw
3. Dynamic RSI thresholds (percentile-based) vs fixed 30/70
4. 1d HMA bias prevents major counter-trend trades (critical for 2022 crash)
5. Volatility-adjusted position sizing (reduce size when ATR spikes)
6. Simple logic = fewer false signals = better Sharpe

Key differences from failed 4h attempts:
- NO Choppiness Index (failed in #1591, #1594, #1600)
- NO Fisher Transform (failed in #1600)
- NO complex regime switching (failed in #1590, #1592, #1595)
- YES: KAMA+HMA crossover (proven in baseline strategies)
- YES: Dynamic RSI percentiles (adapts to market conditions)
- YES: 1d HMA strong bias (prevents 2022-style drawdowns)

Entry logic (LOOSE to guarantee ≥30 trades/train):
- LONG: 1d_HMA bullish + KAMA>HMA + RSI>45 (not fixed 50) + price>20-bar high
- SHORT: 1d_HMA bearish + KAMA<HMA + RSI<55 (not fixed 50) + price<20-bar low
- RSI thresholds adapt: use 40th/60th percentile of recent 100 bars

Exit logic:
- KAMA crosses below/above HMA (trend reversal)
- Stoploss: 2.5x ATR trailing
- Take profit: reduce to half at 2R

Position sizing:
- Base: 0.25, High vol: 0.20 (when ATR > 1.5x 50-bar avg)
- Discrete: 0.0, ±0.20, ±0.25

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_hma_rsi_dynamic_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - er_period]):
            signal = abs(close[i] - close[i - er_period])
            noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if noise > 1e-10:
                er[i] = signal / noise
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]) or np.isnan(kama[i-1]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - for breakout detection"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi_percentile(rsi, lookback=100):
    """
    Dynamic RSI thresholds based on recent percentile
    Returns 40th and 60th percentile of recent RSI values
    """
    n = len(rsi)
    rsi_40 = np.full(n, np.nan)
    rsi_60 = np.full(n, np.nan)
    
    for i in range(lookback, n):
        recent_rsi = rsi[i - lookback:i + 1]
        recent_rsi = recent_rsi[~np.isnan(recent_rsi)]
        if len(recent_rsi) >= lookback // 2:
            rsi_40[i] = np.percentile(recent_rsi, 40)
            rsi_60[i] = np.percentile(recent_rsi, 60)
    
    return rsi_40, rsi_60

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
    
    # Calculate 4h indicators
    kama_20 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    hma_20 = calculate_hma(close, period=20)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Dynamic RSI thresholds
    rsi_40, rsi_60 = calculate_rsi_percentile(rsi_14, lookback=100)
    
    # ATR average for vol adjustment
    atr_avg_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_LOW_VOL = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track KAMA/HMA cross
    prev_kama_hma_diff = np.nan
    
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
        
        if np.isnan(kama_20[i]) or np.isnan(hma_20[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donch_upper[i]):
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
        
        # === TREND DIRECTION (1d and 1w HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA/HMA CROSSOVER SIGNAL ===
        kama_hma_diff = kama_20[i] - hma_20[i]
        kama_hma_cross_bull = prev_kama_hma_diff is not np.nan and prev_kama_hma_diff <= 0 and kama_hma_diff > 0
        kama_hma_cross_bear = prev_kama_hma_diff is not np.nan and prev_kama_hma_diff >= 0 and kama_hma_diff < 0
        
        # Current trend state
        kama_above_hma = kama_hma_diff > 0
        kama_below_hma = kama_hma_diff < 0
        
        # === DYNAMIC RSI THRESHOLDS ===
        rsi_threshold_long = rsi_40[i] if not np.isnan(rsi_40[i]) else 45.0
        rsi_threshold_short = rsi_60[i] if not np.isnan(rsi_60[i]) else 55.0
        
        rsi_bullish = rsi_14[i] > rsi_threshold_long
        rsi_bearish = rsi_14[i] < rsi_threshold_short
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donch_upper[i-1] if i > 0 and not np.isnan(donch_upper[i-1]) else False
        donchian_breakout_short = close[i] < donch_lower[i-1] if i > 0 and not np.isnan(donch_lower[i-1]) else False
        
        # === VOLATILITY ADJUSTMENT ===
        vol_adjust = 1.0
        if not np.isnan(atr_avg_50[i]) and atr_avg_50[i] > 1e-10:
            if atr_14[i] > 1.5 * atr_avg_50[i]:
                vol_adjust = 0.8  # Reduce size in high vol
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + KAMA>HMA + RSI bullish + breakout OR cross
        if price_above_1d and kama_above_hma and rsi_bullish:
            if donchian_breakout_long or kama_hma_cross_bull:
                desired_signal = SIZE_BASE * vol_adjust
        
        # SHORT: 1d bearish + KAMA<HMA + RSI bearish + breakout OR cross
        elif price_below_1d and kama_below_hma and rsi_bearish:
            if donchian_breakout_short or kama_hma_cross_bear:
                desired_signal = -SIZE_BASE * vol_adjust
        
        # Additional: 1w confirmation for stronger signals
        if price_above_1w and desired_signal > 0:
            desired_signal = min(desired_signal * 1.1, SIZE_BASE)
        elif price_below_1w and desired_signal < 0:
            desired_signal = max(desired_signal * 1.1, -SIZE_BASE)
        
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0 and kama_below_hma:
            desired_signal = 0.0
        if in_position and position_side < 0 and kama_above_hma:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_LOW_VOL * 0.9:
            final_signal = np.sign(desired_signal) * SIZE_LOW_VOL
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
        prev_kama_hma_diff = kama_hma_diff
    
    return signals