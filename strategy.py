#!/usr/bin/env python3
"""
Experiment #622: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + Choppiness Regime

Hypothesis: 4h timeframe offers optimal balance between trade frequency (20-50/year)
and signal quality. KAMA (Kaufman Adaptive Moving Average) outperforms HMA/EMA in
choppy markets by adjusting smoothing based on volatility. Combined with Choppiness
Index regime filter and 1d/1w HTF bias, this should beat 6h Fisher strategy (#620).

Key improvements over #620:
1. KAMA instead of HMA - adapts smoothing ratio based on market efficiency
2. Cleaner regime logic - explicit trend vs range modes
3. More generous entry thresholds - ensure 25+ trades/year
4. 4h TF = fewer false signals than 1h, more trades than 6h
5. Simpler stoploss - 2.5x ATR trailing

Strategy logic:
1. 1w HMA(21) = macro trend bias
2. 1d HMA(21) = medium trend bias
3. 4h KAMA(14) = adaptive local trend
4. 4h Choppiness(14) = regime detection
5. 4h RSI(14) = entry timing filter
6. 4h ATR(14) = volatility + stoploss

Regime-adaptive entries:
- TREND (CHOP<45): Follow HTF + KAMA direction, RSI confirms momentum
- RANGE (CHOP>55): Mean revert at RSI extremes (30/70) with HTF filter
- Stoploss: 2.5*ATR trailing from entry

Target: Sharpe>0.40, trades>=30 train, trades>=3 test
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_regime_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    
    Formula:
    1. Efficiency Ratio (ER) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    2. Fast SC = 2/(fast+1), Slow SC = 2/(slow+1)
    3. Smoothing Constant = ER * (Fast SC - Slow SC) + Slow SC
    4. KAMA = KAMA[prev] + SC^2 * (close - KAMA[prev])
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Initialize KAMA at period
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
        else:
            sc = er[i] * (fast_sc - slow_sc) + slow_sc
            kama[i] = kama[i - 1] + (sc ** 2) * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Hull Moving Average - for HTF trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
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
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        
        # === 4H LOCAL TREND (KAMA) ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === KAMA SLOPE (momentum) ===
        kama_slope_bull = False
        kama_slope_bear = False
        if i >= 3 and not np.isnan(kama[i-3]):
            kama_slope_bull = kama[i] > kama[i-3]
            kama_slope_bear = kama[i] < kama[i-3]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean reversion)
        chop_trend = chop[i] < 45.0   # Trending (trend follow)
        
        # === RSI FILTER ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_neutral = 40.0 < rsi[i] < 60.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HTF + KAMA direction
        if chop_trend:
            # Long: HTF bull + KAMA bull + KAMA rising + RSI not overbought
            if htf_bull and kama_bull and kama_slope_bull and rsi[i] < 70:
                desired_signal = SIZE_STRONG
            # Short: HTF bear + KAMA bear + KAMA falling + RSI not oversold
            elif htf_bear and kama_bear and kama_slope_bear and rsi[i] > 30:
                desired_signal = -SIZE_STRONG
            # Weaker signal: KAMA cross with HTF confirmation
            elif htf_bull and kama_bull and rsi_neutral:
                desired_signal = SIZE_BASE
            elif htf_bear and kama_bear and rsi_neutral:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean revert at RSI extremes with HTF filter
        elif chop_range:
            # Long: RSI oversold + HTF not strongly bearish
            if rsi_oversold and not htf_bear:
                desired_signal = SIZE_BASE
            # Short: RSI overbought + HTF not strongly bullish
            elif rsi_overbought and not htf_bull:
                desired_signal = -SIZE_BASE
            # RSI recovery from extreme
            elif rsi[i] < 40 and i > 0 and rsi[i] > rsi[i-1]:
                desired_signal = SIZE_BASE * 0.8
            elif rsi[i] > 60 and i > 0 and rsi[i] < rsi[i-1]:
                desired_signal = -SIZE_BASE * 0.8
        
        # NEUTRAL REGIME: Wait for strong HTF + KAMA alignment
        else:
            if htf_bull and kama_bull and kama_slope_bull:
                desired_signal = SIZE_BASE
            elif htf_bear and kama_bear and kama_slope_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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