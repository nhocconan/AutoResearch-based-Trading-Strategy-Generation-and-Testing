#!/usr/bin/env python3
"""
Experiment #032: 12h Primary + 1d/1w HTF — Choppiness Regime + HMA Trend + RSI

Hypothesis: After 31 experiments, the winning pattern for 12h is:
1. Choppiness Index detects regime (CHOP>61.8=range, CHOP<38.2=trend)
2. 1d HMA provides HTF trend bias (only trade with higher timeframe)
3. 1w HMA provides macro filter (avoid counter-macro trades)
4. In TREND regime: HMA crossover + RSI pullback entries
5. In RANGE regime: RSI extremes + Bollinger mean reversion
6. ATR trailing stop (2.5x) for risk management

Key insight: Dual-regime approach adapts to market conditions.
- Trend regime (CHOP<38.2): Follow HMA direction, enter on RSI pullbacks
- Range regime (CHOP>61.8): Mean revert at Bollinger bands, RSI extremes

Entry Logic (LOOSE for trade generation):
- Long Trend: 1d_HMA_bull + 1w_HMA_bull + CHOP<38.2 + price>HMA21 + RSI<55
- Short Trend: 1d_HMA_bear + 1w_HMA_bear + CHOP<38.2 + price<HMA21 + RSI>45
- Long Range: 1d_HMA_bull + CHOP>61.8 + price<BB_lower + RSI<45
- Short Range: 1d_HMA_bear + CHOP>61.8 + price>BB_upper + RSI>55

Size: 0.30 (discrete, proven safe through 2022 crash)
Target: Sharpe>0.25, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_hma_rsi_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    double_wma_half = 2.0 * wma_half - wma_full
    hma = wma(double_wma_half, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds for trade generation"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - detects trending vs ranging markets
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10 or atr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands - for mean reversion entries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=9)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size - safe through 77% crash
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(hma_12h[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
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
        
        # === HTF BIAS (1d and 1w HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 12h TREND ===
        hma_12h_bull = close[i] > hma_12h[i]
        hma_12h_bear = close[i] < hma_12h[i]
        hma_fast_above_slow = hma_12h_fast[i] > hma_12h[i] if not np.isnan(hma_12h_fast[i]) else False
        hma_fast_below_slow = hma_12h_fast[i] < hma_12h[i] if not np.isnan(hma_12h_fast[i]) else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop[i] < 38.2  # Trending market
        is_range_regime = chop[i] > 61.8  # Ranging market
        
        # === DESIRED SIGNAL (LOOSE thresholds for trade generation) ===
        desired_signal = 0.0
        
        # LONG in TREND regime
        if is_trend_regime and hma_1d_bull and hma_1w_bull:
            if hma_12h_bull and hma_fast_above_slow and rsi[i] < 55.0:
                # Strong uptrend with pullback
                desired_signal = SIZE
            elif hma_12h_bull and rsi[i] < 50.0:
                # Uptrend with deeper pullback
                desired_signal = SIZE
        
        # SHORT in TREND regime
        if is_trend_regime and hma_1d_bear and hma_1w_bear:
            if hma_12h_bear and hma_fast_below_slow and rsi[i] > 45.0:
                # Strong downtrend with rally
                desired_signal = -SIZE
            elif hma_12h_bear and rsi[i] > 50.0:
                # Downtrend with deeper rally
                desired_signal = -SIZE
        
        # LONG in RANGE regime (mean reversion)
        if is_range_regime and hma_1d_bull:
            if close[i] < bb_lower[i] and rsi[i] < 45.0:
                # Price at lower BB with oversold RSI
                desired_signal = SIZE
            elif rsi[i] < 35.0 and hma_12h_bull:
                # Deeply oversold in range
                desired_signal = SIZE
        
        # SHORT in RANGE regime (mean reversion)
        if is_range_regime and hma_1d_bear:
            if close[i] > bb_upper[i] and rsi[i] > 55.0:
                # Price at upper BB with overbought RSI
                desired_signal = -SIZE
            elif rsi[i] > 65.0 and hma_12h_bear:
                # Deeply overbought in range
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals