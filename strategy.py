#!/usr/bin/env python3
"""
Experiment #548: 4h Primary + 1d HTF — Simplified Regime + RSI Pullback

Hypothesis: Previous 4h strategies failed due to TOO MANY confluence filters
(ADX + CHOP + HMA + KAMA + RSI all must agree = 0 trades). This version:
1. Uses ONLY ADX for regime (ADX>25=trend, ADX<20=range) - simpler
2. 1d HMA for trend bias only (not both 1d AND 1w)
3. 4h HMA for entry timing
4. RSI pullback entries (less strict: RSI<45 for long, RSI>55 for short)
5. Fewer filters = MORE trades (target 80-200 trades over 4 years)

Key insight from failures: #536-547 all got Sharpe=0.000 because entry conditions
were too strict. This strategy LOOSENS entries while keeping risk management.

Strategy logic:
1. 1d HMA(21) = trend bias (price > HMA = bull bias, price < HMA = bear bias)
2. 4h ADX(14) = regime (ADX>25 = trend follow, ADX<20 = mean revert)
3. 4h HMA(16) = entry trigger (crossover + pullback)
4. 4h RSI(14) = entry timing (oversold in uptrend, overbought in downtrend)
5. ATR(14)*2.5 stoploss on all positions

Entry rules (LOOSENED for trade generation):
- TREND (ADX>25): Long if price>1d_HMA AND price>4h_HMA AND RSI<55
- TREND (ADX>25): Short if price<1d_HMA AND price<4h_HMA AND RSI>45
- RANGE (ADX<20): Long if RSI<35, Short if RSI>65

Target: Sharpe>0.40, trades>=80 train, trades>=10 test, ALL symbols Sharpe>0
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simplified_regime_rsi_hma_1d_v1"
timeframe = "4h"
leverage = 1.0

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
    Average Directional Index - measures trend strength
    ADX > 25 = trending market
    ADX < 20 = ranging market
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
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(close, period=16)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = -0.30
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(hma_4h[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
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
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND (HMA) ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # HMA slope
        hma_slope_bull = hma_4h[i] > hma_4h[i-3] if i >= 3 and not np.isnan(hma_4h[i-3]) else False
        hma_slope_bear = hma_4h[i] < hma_4h[i-3] if i >= 3 and not np.isnan(hma_4h[i-3]) else False
        
        # === ADX REGIME ===
        is_trend = adx[i] > 25.0
        is_range = adx[i] < 20.0
        
        # === RSI LEVELS (LOOSENED for more trades) ===
        rsi_oversold = rsi[i] < 45.0
        rsi_overbought = rsi[i] > 55.0
        rsi_extreme_oversold = rsi[i] < 35.0
        rsi_extreme_overbought = rsi[i] > 65.0
        
        # === ENTRY LOGIC (SIMPLIFIED) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HTF bias with 4h confirmation
        if is_trend:
            # Long: HTF bull + 4h bull + RSI not overbought
            if htf_bull and hma_bull and rsi_oversold:
                desired_signal = SIZE_LONG
            # Short: HTF bear + 4h bear + RSI not oversold
            elif htf_bear and hma_bear and rsi_overbought:
                desired_signal = SIZE_SHORT
            # HMA crossover entries (more frequent)
            elif htf_bull and hma_bull and hma_slope_bull:
                desired_signal = SIZE_LONG
            elif htf_bear and hma_bear and hma_slope_bear:
                desired_signal = SIZE_SHORT
        
        # RANGE REGIME: Mean reversion at RSI extremes
        elif is_range:
            if rsi_extreme_oversold:
                desired_signal = SIZE_LONG
            elif rsi_extreme_overbought:
                desired_signal = SIZE_SHORT
        
        # TRANSITION (20 <= ADX <= 25): Reduced size, HTF bias only
        else:
            if htf_bull and rsi_oversold:
                desired_signal = SIZE_HALF
            elif htf_bear and rsi_overbought:
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
        if desired_signal >= SIZE_LONG * 0.9:
            final_signal = SIZE_LONG
        elif desired_signal <= SIZE_SHORT * 0.9:
            final_signal = SIZE_SHORT
        elif desired_signal >= SIZE_HALF * 0.9:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.9:
            final_signal = -SIZE_HALF
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