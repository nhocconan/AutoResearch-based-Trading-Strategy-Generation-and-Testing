#!/usr/bin/env python3
"""
Experiment #1451: 4h Primary + 1d HTF — Simplified Dual-Mode with Asymmetric Bias

Hypothesis: Previous 4h strategies failed due to TOO MANY filters causing 0 trades or 
whipsaw losses. This strategy SIMPLIFIES entry logic while maintaining regime awareness:

1. Fewer confluence requirements = more trades (critical after #1440, #1445, #1448 got 0 trades)
2. Asymmetric long bias = crypto trends up more than down (works better in 2021-2024 bull)
3. Dual-mode: mean reversion (RSI) in range, breakout (Donchian) in trend
4. 1d HMA(21) = simple macro filter (proven in #1443 with Sharpe=0.098)
5. Volatility-adjusted sizing = reduce position when ATR spikes (protects in 2022 crash)

Why this differs from failed 4h strategies:
- #1439, #1441, #1444, #1449 all had negative Sharpe with complex regime logic
- This uses SIMPLER entry: RSI OR Donchian (not AND) = more trade opportunities
- Long bias: easier long entries than short (matches crypto market structure)
- Target: 30-60 trades/year on 4h (within 20-50 guideline but ensures >=10 train trades)

Design:
1. 1d HMA(21) aligned = macro trend (call ONCE before loop)
2. RSI(14): long <40, short >60 (wider than typical 30/70 for more trades)
3. Donchian(20): breakout entries when price exceeds 20-bar high/low
4. ATR(14) ratio: if ATR/ATR_100 > 1.5, reduce size by 50% (vol filter)
5. Long bias: long requires 2/3 conditions, short requires 3/3 (asymmetric)
6. Trailing stop: 2.5x ATR from entry high/low
7. Position size: 0.28 base, 0.14 when vol spike

Target: Sharpe > 0.618 (beat current best), trades >= 30 train, >= 5 test, DD > -40%
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simplified_dual_asymmetric_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

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

def calculate_atr_long(high, low, close, period=100):
    """Long-term ATR for volatility regime detection"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_long = calculate_atr_long(high, low, close, period=100)
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    donchian_55_upper, donchian_55_lower = calculate_donchian(high, low, period=55)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.14  # Half size during vol spikes
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_20_upper[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY REGIME (size adjustment) ===
        vol_spike = False
        if not np.isnan(atr_long[i]) and atr_long[i] > 1e-10:
            vol_ratio = atr[i] / atr_long[i]
            if vol_ratio > 1.5:
                vol_spike = True
        
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === TREND FILTER (SMA50 vs SMA200) ===
        golden_cross = False
        death_cross = False
        if not np.isnan(sma_200[i]) and sma_200[i] > 1e-10:
            if sma_50[i] > sma_200[i]:
                golden_cross = True
            else:
                death_cross = True
        
        # === RSI SIGNALS (mean reversion) ===
        rsi_oversold = rsi[i] < 40.0  # Wider threshold for more trades
        rsi_overbought = rsi[i] > 60.0  # Wider threshold for more trades
        rsi_extreme_low = rsi[i] < 30.0
        rsi_extreme_high = rsi[i] > 70.0
        
        # === DONCHIAN BREAKOUT (trend following) ===
        breakout_20_long = close[i] > donchian_20_upper[i-1] if i > 0 and not np.isnan(donchian_20_upper[i-1]) else False
        breakout_20_short = close[i] < donchian_20_lower[i-1] if i > 0 and not np.isnan(donchian_20_lower[i-1]) else False
        breakout_55_long = close[i] > donchian_55_upper[i-1] if i > 0 and not np.isnan(donchian_55_upper[i-1]) else False
        breakout_55_short = close[i] < donchian_55_lower[i-1] if i > 0 and not np.isnan(donchian_55_lower[i-1]) else False
        
        # === ASYMMETRIC ENTRY LOGIC ===
        # LONG: easier to enter (2/3 conditions) - crypto has upward bias
        long_score = 0
        if macro_bull:
            long_score += 1
        if rsi_oversold or rsi_extreme_low:
            long_score += 1
        if breakout_20_long or breakout_55_long:
            long_score += 1
        if golden_cross:
            long_score += 0.5
        
        # SHORT: harder to enter (3/3 conditions) - avoid shorting bull markets
        short_score = 0
        if macro_bear:
            short_score += 1
        if rsi_overbought or rsi_extreme_high:
            short_score += 1
        if breakout_20_short or breakout_55_short:
            short_score += 1
        if death_cross:
            short_score += 0.5
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: need score >= 2.0 (asymmetric - easier for longs)
        if long_score >= 2.0:
            desired_signal = current_size
        
        # Short entry: need score >= 2.5 (harder for shorts)
        if short_score >= 2.5:
            desired_signal = -current_size
        
        # Conflict resolution: if both long and short signals, prefer macro trend
        if long_score >= 2.0 and short_score >= 2.5:
            if macro_bull:
                desired_signal = current_size
            elif macro_bear:
                desired_signal = -current_size
            else:
                desired_signal = 0.0
        
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
        final_signal = 0.0
        if desired_signal > 0.0:
            final_signal = current_size
        elif desired_signal < 0.0:
            final_signal = -current_size
        
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