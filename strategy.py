#!/usr/bin/env python3
"""
Experiment #615: 6h Primary + 12h/1d HTF — Simplified Trend Pullback with Regime Filter

Hypothesis: Previous 6h strategies failed due to overly complex entry conditions resulting
in zero or few trades. This strategy uses SIMPLER logic with looser entry thresholds to
ensure sufficient trade frequency while maintaining HTF confirmation for quality.

Key changes from failed experiments:
1. SIMPLER entry logic - RSI pullback within trend (not extreme values)
2. LOOSER RSI thresholds - 35-45 for long, 55-65 for short (vs 25/75 extremes)
3. 12h ADX as trend strength filter (ADX>18 = trend valid, lower than typical 25)
4. 1d HMA(21) for macro trend bias only (single HTF filter)
5. No choppiness index - reduces complexity and potential for zero trades
6. Asymmetric sizing - full size when HTF aligned, half when neutral

Strategy logic:
1. 1d HMA(21) = macro trend (price above = bull bias, below = bear bias)
2. 12h ADX(14) = trend strength (ADX>18 = trend, ADX<18 = range)
3. 6h RSI(14) = entry timing (pullback entries within trend)
4. 6h HMA(9) = short-term momentum confirmation
5. 6h ATR(14)*2.5 = stoploss on all positions

Entry rules (LOOSE to ensure trades):
- LONG: 1d HMA bull + 12h ADX>18 + 6h RSI 35-50 + price>6h HMA
- SHORT: 1d HMA bear + 12h ADX>18 + 6h RSI 50-65 + price<6h HMA
- RANGE: 12h ADX<18 + RSI extremes (25/75) for mean reversion

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_trend_pullback_rsi_12h1d_simple_v1"
timeframe = "6h"
leverage = 1.0

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
    """Hull Moving Average - faster response than EMA"""
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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - trend strength indicator
    ADX > 25 = strong trend, ADX < 20 = range
    Using lower threshold (18) for 6h to ensure more signals
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    # Smooth with Wilder's method
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 12h ADX for trend strength
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate 6h indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_6h = calculate_hma(close, period=9)
    
    signals = np.zeros(n)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    SIZE_QUARTER = 0.10
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d macro trend) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h ADX TREND STRENGTH ===
        adx_trend = adx_12h_aligned[i] > 18.0  # Lower threshold for more signals
        adx_range = adx_12h_aligned[i] < 18.0
        
        # === 6h MOMENTUM ===
        mom_bull = close[i] > hma_6h[i]
        mom_bear = close[i] < hma_6h[i]
        
        # === RSI LEVELS (LOOSE thresholds for more trades) ===
        rsi_long_zone = 35.0 <= rsi[i] <= 50.0  # Pullback in uptrend
        rsi_short_zone = 50.0 <= rsi[i] <= 65.0  # Pullback in downtrend
        rsi_extreme_long = rsi[i] < 30.0
        rsi_extreme_short = rsi[i] > 70.0
        
        # === ENTRY LOGIC (SIMPLIFIED for trade frequency) ===
        desired_signal = 0.0
        
        # TREND REGIME: Pullback entries with HTF confirmation
        if adx_trend:
            # Long: 1d bull + RSI pullback + 6h momentum confirm
            if htf_bull and rsi_long_zone and mom_bull:
                desired_signal = SIZE_FULL
            # Short: 1d bear + RSI pullback + 6h momentum confirm
            elif htf_bear and rsi_short_zone and mom_bear:
                desired_signal = -SIZE_FULL
            # Weaker signal when HTF neutral but 6h momentum strong
            elif rsi_extreme_long and mom_bull:
                desired_signal = SIZE_HALF
            elif rsi_extreme_short and mom_bear:
                desired_signal = -SIZE_HALF
        
        # RANGE REGIME: Mean reversion at extremes
        elif adx_range:
            if rsi_extreme_long:
                desired_signal = SIZE_HALF
            elif rsi_extreme_short:
                desired_signal = -SIZE_HALF
        
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
        if desired_signal >= SIZE_FULL * 0.9:
            final_signal = SIZE_FULL
        elif desired_signal <= -SIZE_FULL * 0.9:
            final_signal = -SIZE_FULL
        elif desired_signal >= SIZE_HALF * 0.9:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.9:
            final_signal = -SIZE_HALF
        elif abs(desired_signal) >= SIZE_QUARTER * 0.9:
            final_signal = np.sign(desired_signal) * SIZE_QUARTER
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