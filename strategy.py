#!/usr/bin/env python3
"""
Experiment #527: 6h Primary + 1d HTF — KAMA Trend + Donchian Breakout + Choppiness Regime

Hypothesis: 6h timeframe sits between 4h and 12h - captures multi-day trends without
excessive noise. KAMA adapts to volatility (fast in trends, slow in chop). Donchian(20)
breakout confirms momentum. Choppiness filters regime. 1d HMA provides macro bias.

Key differences from failed #523 (6h_kama_adx_rsi):
1. Donchian breakout instead of ADX - catches momentum moves ADX misses
2. Looser RSI thresholds (35/65 vs 25/75) - ensures we get trades
3. Simpler regime logic - trend follow when CHOP<50, mean revert when CHOP>60
4. 1d HMA only (not 1w) - 1w too slow for 6h entries
5. Size=0.28 base, 0.32 strong - discrete levels to reduce fee churn

Strategy logic:
1. 1d HMA(21) = macro trend bias (call ONCE before loop)
2. 6h KAMA(10,2,30) = adaptive trend following
3. 6h Donchian(20) = breakout momentum confirmation
4. 6h Choppiness(14) = regime filter (CHOP<50=trend, CHOP>60=range)
5. 6h RSI(14) = entry timing (35/65 thresholds for trade frequency)
6. ATR(14)*2.5 stoploss on all positions

Entry conditions (LOOSENED to ensure ≥30 trades):
- TREND: Price>KAMA + KAMA slope + Donchian breakout + HTF alignment
- RANGE: RSI extreme + price near Donchian bounds
- Size: 0.28 base, 0.32 strong confirmation

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_donchian_chop_regime_1d_v1"
timeframe = "6h"
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

def calculate_donchian(high, low, period=20):
    """
    Donchian Channels - breakout system
    Returns: upper_channel, lower_channel, mid_channel
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    mid = (upper + lower) / 2.0
    
    return upper, lower, mid

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
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.28
    SIZE_STRONG = 0.32
    
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
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
        
        # === HTF BIAS (1d macro) ===
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
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # Price position within Donchian channel
        channel_range = donchian_upper[i] - donchian_lower[i]
        if channel_range > 1e-10:
            channel_position = (close[i] - donchian_lower[i]) / channel_range
        else:
            channel_position = 0.5
        
        near_upper = channel_position > 0.75
        near_lower = channel_position < 0.25
        
        # === CHOPPINESS REGIME ===
        chop_trend = chop[i] < 50.0   # Trending market
        chop_range = chop[i] > 60.0   # Range-bound market
        chop_neutral = not chop_trend and not chop_range
        
        # === RSI EXTREMES (LOOSENED for trade frequency) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_extreme_oversold = rsi[i] < 30.0
        rsi_extreme_overbought = rsi[i] > 70.0
        
        # RSI momentum
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 and not np.isnan(rsi[i-1]) else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 and not np.isnan(rsi[i-1]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow KAMA + Donchian breakout + HTF alignment
        if chop_trend:
            # Strong long: HTF bull + KAMA bull + slope + Donchian breakout
            if htf_bull and kama_bull and kama_slope_bull and donchian_breakout_long:
                desired_signal = SIZE_STRONG
            # Strong short: HTF bear + KAMA bear + slope + Donchian breakout
            elif htf_bear and kama_bear and kama_slope_bear and donchian_breakout_short:
                desired_signal = -SIZE_STRONG
            # Base long: HTF bull + KAMA bull + RSI not overbought
            elif htf_bull and kama_bull and kama_slope_bull and not rsi_overbought:
                desired_signal = SIZE_BASE
            # Base short: HTF bear + KAMA bear + RSI not oversold
            elif htf_bear and kama_bear and kama_slope_bear and not rsi_oversold:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at channel bounds + RSI extremes
        elif chop_range:
            # Long at lower band + oversold RSI
            if near_lower and rsi_extreme_oversold and rsi_rising:
                desired_signal = SIZE_BASE
            # Short at upper band + overbought RSI
            elif near_upper and rsi_extreme_overbought and rsi_falling:
                desired_signal = -SIZE_BASE
            # Recovery from extreme oversold
            elif rsi_extreme_oversold and rsi_rising and htf_bull:
                desired_signal = SIZE_BASE * 0.8
            # Recovery from extreme overbought
            elif rsi_extreme_overbought and rsi_falling and htf_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # NEUTRAL REGIME: Reduced size, wait for confirmation
        elif chop_neutral:
            # Wait for Donchian breakout with HTF alignment
            if htf_bull and donchian_breakout_long and kama_bull:
                desired_signal = SIZE_BASE * 0.7
            elif htf_bear and donchian_breakout_short and kama_bear:
                desired_signal = -SIZE_BASE * 0.7
            # RSI recovery plays
            elif htf_bull and rsi_extreme_oversold and rsi_rising:
                desired_signal = SIZE_BASE * 0.6
            elif htf_bear and rsi_extreme_overbought and rsi_falling:
                desired_signal = -SIZE_BASE * 0.6
        
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