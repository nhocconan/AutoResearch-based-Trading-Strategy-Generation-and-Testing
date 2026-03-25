#!/usr/bin/env python3
"""
Experiment #1566: 1d Primary + 1w HTF — Regime-Switching KAMA Strategy

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance
of trade frequency (20-40/year) and signal quality. Key innovations:

1. 1w HMA(21) for MAJOR regime (bull/bear) - only trade WITH weekly trend
2. 1d KAMA(14) for adaptive trend following - KAMA reduces whipsaw in chop
3. 1d RSI(7) for fast mean reversion - faster than RSI(14), catches dips
4. 1d Choppiness(14) for entry timing - avoid entries in extreme chop
5. Asymmetric logic: LONG only when 1w bullish, SHORT only when 1w bearish
6. ATR(14) 3x trailing stop for risk management

Why this differs from failed CRSI+Chop strategies:
- KAMA adapts to volatility (ER parameter) vs static HMA
- RSI(7) faster response than RSI(14) for daily entries
- Regime filter is 1w HMA slope, not Choppiness (more stable)
- Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Entry logic (LOOSE to guarantee trades):
- LONG: 1w_HMA rising + 1d_KAMA bullish + RSI(7)<35 + CHOP<65
- SHORT: 1w_HMA falling + 1d_KAMA bearish + RSI(7)>65 + CHOP<65

Target: Sharpe>0.6, trades>=20 train, trades>=3 test, DD>-35%
Timeframe: 1d
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi7_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    ER = 1 in strong trend, ER = 0 in choppy market
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        if price_change < 1e-10:
            er[i] = 0.0
            continue
        vol_sum = 0.0
        for j in range(i - period + 1, i + 1):
            vol_sum += abs(close[j] - close[j - 1])
        if vol_sum > 1e-10:
            er[i] = price_change / vol_sum
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
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

def calculate_rsi(close, period=7):
    """Relative Strength Index - faster period for daily entries"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Also get 1w close for slope calculation
    close_1w_raw = df_1w['close'].values
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w_raw)
    
    # Calculate 1d indicators
    kama_14 = calculate_kama(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        if np.isnan(rsi_7[i]) or np.isnan(kama_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(close_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY REGIME (1w HMA slope + price position) ===
        # Calculate 1w HMA slope (compare to 3 bars ago)
        hma_1w_slope = 0.0
        if i >= 3 and not np.isnan(hma_1w_aligned[i-3]):
            hma_1w_slope = (hma_1w_aligned[i] - hma_1w_aligned[i-3]) / hma_1w_aligned[i-3]
        
        weekly_bullish = hma_1w_slope > 0.001 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_slope < -0.001 and close[i] < hma_1w_aligned[i]
        weekly_neutral = not weekly_bullish and not weekly_bearish
        
        # === DAILY KAMA TREND ===
        kama_bullish = close[i] > kama_14[i]
        kama_bearish = close[i] < kama_14[i]
        
        # === RSI (7) for fast mean reversion ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 35
        rsi_overbought = rsi > 65
        
        # === CHOPPINESS FILTER ===
        chop = chop_14[i]
        not_extreme_chop = chop < 70  # avoid entries in extreme chop
        
        # === ENTRY LOGIC (ASYMMETRIC - trade with weekly trend) ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + Daily KAMA bullish + RSI oversold + Not extreme chop
        if weekly_bullish and kama_bullish and rsi_oversold and not_extreme_chop:
            desired_signal = SIZE_STRONG
        
        # LONG weaker: Weekly bullish + KAMA bullish + RSI not overbought
        elif weekly_bullish and kama_bullish and rsi < 60 and not_extreme_chop:
            desired_signal = SIZE_BASE
        
        # SHORT: Weekly bearish + Daily KAMA bearish + RSI overbought + Not extreme chop
        elif weekly_bearish and kama_bearish and rsi_overbought and not_extreme_chop:
            desired_signal = -SIZE_STRONG
        
        # SHORT weaker: Weekly bearish + KAMA bearish + RSI not oversold
        elif weekly_bearish and kama_bearish and rsi > 40 and not_extreme_chop:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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