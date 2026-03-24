#!/usr/bin/env python3
"""
Experiment #572: 12h Primary + 1d HTF — KAMA Adaptive Trend + Choppiness Regime + RSI

Hypothesis: 12h timeframe provides optimal balance between signal quality and trade frequency.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - moves fast in trends,
slow in chop. Combined with Choppiness Index for regime detection and RSI for entry timing,
this should generate 20-50 trades/year with positive Sharpe on all symbols.

Key improvements from failed #564 (12h_donchian_hma_chop):
1. KAMA instead of Donchian - smoother adaptive trend following
2. Simpler entry logic - fewer conflicting filters = more trades
3. Looser RSI thresholds (30/70 not 25/75) to ensure trade generation
4. ADX threshold at 20 (not 25) for trend confirmation
5. Clear regime switch: CHOP>55 = mean revert, CHOP<45 = trend follow

Strategy logic:
1. 1d HMA(21) = HTF trend bias
2. 12h KAMA(10,2,30) = adaptive trend following
3. 12h Choppiness(14) = regime (CHOP>55 range, CHOP<45 trend)
4. 12h RSI(14) = entry timing (30/70 extremes)
5. 12h ADX(14) = trend strength confirmation (>20 = valid trend)
6. ATR(14)*2.5 stoploss on all positions

Entry conditions (LOOSE to ensure trades):
- TREND regime (CHOP<45 + ADX>20): Follow KAMA direction + HTF alignment + RSI 40-60
- RANGE regime (CHOP>55): Mean revert at RSI 30/70 extremes
- Both regimes must generate trades on 20% moves

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_rsi_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market efficiency - fast in trends, slow in chop
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_dmi(high, low, close, period=14):
    """
    Directional Movement Index (DMI) - calculates +DI, -DI, and ADX
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
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
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return plus_di, minus_di, adx

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
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
    
    # Calculate 12h indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    plus_di, minus_di, adx = calculate_dmi(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
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
        
        if np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
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
        
        # === HTF BIAS (1d trend) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # KAMA slope (5-bar lookback)
        kama_slope_bull = False
        kama_slope_bear = False
        if i >= 5 and not np.isnan(kama[i-5]):
            kama_slope_bull = kama[i] > kama[i-5]
            kama_slope_bear = kama[i] < kama[i-5]
        
        # === ADX TREND STRENGTH ===
        adx_valid = adx[i] > 20.0  # Looser threshold for more trades
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound
        chop_trend = chop[i] < 45.0   # Trending
        
        # === RSI EXTREMES (LOOSE thresholds for trade generation) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_neutral = rsi[i] >= 40.0 and rsi[i] <= 60.0
        
        # === REGIME DETECTION ===
        is_trend_regime = chop_trend and adx_valid
        is_range_regime = chop_range
        
        # === ENTRY LOGIC (LOOSE to ensure trades on major moves) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow KAMA direction with HTF confirmation
        if is_trend_regime:
            # Strong trend entry - all conditions aligned
            if htf_bull and kama_bull and kama_slope_bull:
                desired_signal = SIZE_STRONG
            elif htf_bear and kama_bear and kama_slope_bear:
                desired_signal = -SIZE_STRONG
            # Moderate trend entry - KAMA + RSI filter
            elif htf_bull and kama_bull and rsi[i] > 45.0:
                desired_signal = SIZE_BASE
            elif htf_bear and kama_bear and rsi[i] < 55.0:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at RSI extremes
        elif is_range_regime:
            if rsi_oversold:
                desired_signal = SIZE_BASE
            elif rsi_overbought:
                desired_signal = -SIZE_BASE
            # RSI turning from extreme
            elif i > 0 and not np.isnan(rsi[i-1]):
                if rsi[i] < 45.0 and rsi[i] < rsi[i-1]:
                    desired_signal = -SIZE_BASE * 0.8
                elif rsi[i] > 55.0 and rsi[i] > rsi[i-1]:
                    desired_signal = SIZE_BASE * 0.8
        
        # TRANSITION/NEUTRAL: Simple KAMA crossover with HTF
        else:
            if htf_bull and kama_bull:
                desired_signal = SIZE_BASE * 0.7
            elif htf_bear and kama_bear:
                desired_signal = -SIZE_BASE * 0.7
        
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