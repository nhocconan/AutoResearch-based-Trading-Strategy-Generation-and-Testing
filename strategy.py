#!/usr/bin/env python3
"""
Experiment #1504: 4h Primary + 12h HTF — KAMA Trend + ADX + Choppiness Regime Switch

Hypothesis: After analyzing 1100+ failed strategies, the pattern is clear:
1. Complex dual-regime (chop + crsi) fails on 4h - too many conflicting filters
2. KAMA (Kaufman Adaptive) outperforms HMA/EMA in crypto volatility regimes
3. ADX > 25 confirms trend strength, ADX < 20 = range (mean revert)
4. Choppiness Index > 61.8 = range mode, < 38.2 = trend mode (research-backed)
5. 12h HTF is more responsive than 1d for 4h entries (better trade frequency)
6. Simpler entry logic with fewer confluence requirements = MORE trades

Key design choices:
- KAMA(10) adapts to volatility automatically (no lag in trends, smooth in chop)
- ADX(14) > 25 for trend entries, ADX < 20 for mean-reversion entries
- Choppiness(14) regime switch: trend-follow when CHOP < 38.2, mean-revert when > 61.8
- 12h KAMA for macro bias (more responsive than 1d)
- Position size 0.30 with discrete levels (0.0, ±0.25, ±0.30)
- ATR(14) 2.5x trailing stop for crash protection
- Target: 30-50 trades/year on 4h (optimal fee/trade balance)

Timeframe: 4h (as required)
HTF: 12h (get_htf_data ONCE before loop!)
Position Size: 0.30 (discrete levels)
Target: Sharpe > 0.618 (beat current best), 40-80 trades/train, 8-15 trades/test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_chop_regime_12h_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise: smooth in choppy markets, responsive in trends
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    sc = np.full(n, np.nan)
    mask = ~np.isnan(er)
    sc[mask] = (er[mask] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction). ADX > 25 = strong trend.
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    # Smooth TR, +DM, -DM over period
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX and ADX
    for i in range(period * 2, n):
        if plus_di[i] + minus_di[i] > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending. 
    CHOP > 61.8 = range-bound, CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    # Calculate ATR for each bar (simplified: just true range)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period - 1, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands for mean-reversion entries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF KAMA for trend bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
        if np.isnan(rsi[i]) or np.isnan(kama_4h[i]) or np.isnan(kama_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(adx[i]) or np.isnan(chop[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (12h KAMA) - direction bias ONLY ===
        daily_bull = close[i] > kama_12h_aligned[i]
        daily_bear = close[i] < kama_12h_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        k4_bull = close[i] > kama_4h[i]
        k4_bear = close[i] < kama_4h[i]
        
        # === KAMA SLOPE (trend direction) ===
        kama_slope_bull = kama_4h[i] > kama_4h[i-5] if not np.isnan(kama_4h[i-5]) else False
        kama_slope_bear = kama_4h[i] < kama_4h[i-5] if not np.isnan(kama_4h[i-5]) else False
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx[i] > 25.0
        weak_trend = adx[i] < 20.0
        
        # === CHOPPINESS REGIME ===
        choppy_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        
        # === BOLLINGER POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.002  # within 0.2% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.998  # within 0.2% of upper band
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING MARKET (CHOP < 38.2, ADX > 25)
        # Follow the trend with KAMA alignment
        if trending_market and strong_trend:
            # LONG: 12h bull + 4h bull + KAMA slope up + RSI not overbought
            if daily_bull and k4_bull and kama_slope_bull and rsi[i] < 70.0:
                desired_signal = BASE_SIZE
            # SHORT: 12h bear + 4h bear + KAMA slope down + RSI not oversold
            elif daily_bear and k4_bear and kama_slope_bear and rsi[i] > 30.0:
                desired_signal = -BASE_SIZE
        
        # REGIME 2: CHOPPY MARKET (CHOP > 61.8, ADX < 20)
        # Mean-reversion at Bollinger bands
        elif choppy_market and weak_trend:
            # LONG: Near BB lower + 12h neutral/bull + RSI oversold
            if near_bb_lower and rsi[i] < 35.0:
                desired_signal = BASE_SIZE * 0.7
            # SHORT: Near BB upper + 12h neutral/bear + RSI overbought
            elif near_bb_upper and rsi[i] > 65.0:
                desired_signal = -BASE_SIZE * 0.7
        
        # REGIME 3: TRANSITION (38.2 <= CHOP <= 61.8)
        # Use KAMA crossover with 12h bias
        else:
            # LONG: 12h bull + 4h KAMA cross above + RSI support
            if daily_bull and k4_bull and rsi[i] > 45.0 and rsi[i] < 65.0:
                desired_signal = BASE_SIZE * 0.8
            # SHORT: 12h bear + 4h KAMA cross below + RSI resistance
            elif daily_bear and k4_bear and rsi[i] > 35.0 and rsi[i] < 55.0:
                desired_signal = -BASE_SIZE * 0.8
        
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
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.6
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