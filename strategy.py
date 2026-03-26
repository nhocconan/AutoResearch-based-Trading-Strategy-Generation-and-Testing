#!/usr/bin/env python3
"""
Experiment #004: 1d KAMA + RSI + Choppiness Regime with 1w Trend

HYPOTHESIS: Adaptive KAMA catches trend changes without lag. RSI extremes
provide high-probability mean-reversion entries. Choppiness filter eliminates
whipsaws in range-bound markets. 1w trend alignment ensures trades align with
institutional flow.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- KAMA adapts to volatility regime (fast in trends, slow in chop)
- RSI extremes work in both directions (oversold bounce, overbought dump)
- Weekly trend filter prevents fighting major direction
- Choppiness keeps us out of low-probability range markets

TARGET: 50-100 total trades over 4 years (proven pattern from DB).
DB reference: mtf_1d_kama_rsi_chop_regime_1w_v1 (Sharpe=1.31 on SOL test)

KEY DESIGN:
1. 1w HMA for trend bias (only trade in trend direction)
2. 1d KAMA for adaptive trend signal
3. RSI(14) extremes for entry timing (long <30, short >70)
4. Choppiness < 50 for regime confirmation
5. ATR-based stoploss (2x ATR)
6. Signal: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=14, fast_ema=2, slow_ema=30):
    """
    Kaufman Adaptive Moving Average
    ER = |change| / |volatility|
    fast = 2/(fast_ema+1), slow = 2/(slow_ema+1)
    smoothing = ER*(fast-slow) + slow
    KAMA = prev_KAMA + smoothing*(price - prev_KAMA)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate ER (Efficiency Ratio)
    change = np.abs(close[period:] - close[:-period])
    
    volatility = np.zeros(n - period)
    for i in range(n - period):
        sum_val = 0.0
        for j in range(period):
            sum_val += abs(close[i + j + 1] - close[i + j])
        volatility[i] = sum_val
    
    er = np.zeros(n, dtype=np.float64)
    er[period:] = change / np.where(volatility > 1e-10, volatility, 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast = 2.0 / (fast_ema + 1)
    slow = 2.0 / (slow_ema + 1)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast - slow) + slow) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gains).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging, CHOP < 50 = trending (allow trades)
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

def calculate_hma(close, period):
    """Hull Moving Average"""
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
    
    # Load 1w data for major trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    kama_14 = calculate_kama(close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # KAMA trend direction (rate of change)
    kama_roc = np.zeros(n, dtype=np.float64)
    for i in range(10, n):
        if not np.isnan(kama_14[i]) and not np.isnan(kama_14[i-10]) and kama_14[i-10] > 1e-10:
            kama_roc[i] = (kama_14[i] - kama_14[i-10]) / kama_14[i-10]
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup for slow indicators
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kama_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND REGIME (1w HMA) ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_aligned[i]
        
        # === KAMA DIRECTION ===
        kama_bullish = kama_roc[i] > 0.001  # KAMA rising
        kama_bearish = kama_roc[i] < -0.001  # KAMA falling
        
        # === CHOPPINESS REGIME ===
        is_trending = chop_14[i] < 50.0  # Only trade in trending markets
        
        # === RSI MOMENTUM ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        rsi_neutral = 30 <= rsi <= 70
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Oversold RSI + bullish KAMA + price above 1w HMA + trending
        if is_trending and price_above_1w_hma and kama_bullish:
            if rsi_oversold:
                desired_signal = SIZE
            elif rsi < 40 and not in_position:  # Moderate oversold
                desired_signal = SIZE * 0.5
        
        # SHORT: Overbought RSI + bearish KAMA + price below 1w HMA + trending
        if is_trending and price_below_1w_hma and kama_bearish:
            if rsi_overbought:
                desired_signal = -SIZE
            elif rsi > 60 and not in_position:  # Moderate overbought
                desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (2x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT at 3x ATR from entry ===
        tp_triggered = False
        if in_position and position_side > 0:
            profit_target = entry_price + 3.0 * entry_atr
            if high[i] >= profit_target:
                tp_triggered = True
        
        if in_position and position_side < 0:
            profit_target = entry_price - 3.0 * entry_atr
            if low[i] <= profit_target:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals