#!/usr/bin/env python3
"""
EXPERIMENT #013 - MTF KAMA+Stoch+ADX (1h+4h+1d v1)
==================================================================================================
Hypothesis: Use 1h primary timeframe with 4h KAMA trend + 1h Stochastic entries + 1d ADX regime.
This differs from current best by:
- 1h primary instead of 15m (fewer trades, less noise)
- KAMA adaptive MA instead of Supertrend (better in ranging markets)
- Stochastic oscillator instead of RSI (different momentum signal)
- ADX regime filter instead of BBW (trend strength vs volatility)
- Three timeframes: 1h base, 4h trend, 1d regime

Why this should work:
- KAMA adapts to market volatility (ER-based), reduces whipsaws in chop
- Stochastic gives clearer overbought/oversold signals than RSI
- ADX > 20 on daily ensures we only trade when there's actual trend
- 1h timeframe balances signal frequency with noise reduction
- Conservative position sizing (0.25-0.35) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_stoch_adx_1h_4h_1d_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    change = np.zeros(n)
    volatility = np.zeros(n)
    
    for i in range(er_period, n):
        change[i] = abs(close[i] - close[i - er_period])
        vol_sum = 0.0
        for j in range(1, er_period + 1):
            vol_sum += abs(close[i - j + 1] - close[i - j])
        volatility[i] = vol_sum if vol_sum > 0 else 0.0001
    
    er = change / volatility
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator (%K and %D)"""
    n = len(close)
    if n < k_period + d_period:
        return np.zeros(n), np.zeros(n)
    
    lowest_low = pd.Series(low).rolling(window=k_period, min_periods=k_period).min().values
    highest_high = pd.Series(high).rolling(window=k_period, min_periods=k_period).max().values
    
    k_percent = np.zeros(n)
    for i in range(n):
        if highest_high[i] - lowest_low[i] > 0:
            k_percent[i] = 100.0 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
        else:
            k_percent[i] = 50.0
    
    d_percent = pd.Series(k_percent).rolling(window=d_period, min_periods=d_period).mean().values
    
    return k_percent, d_percent


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    stoch_k_1h, stoch_d_1h = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    
    # Get 4h data using mtf_data helper for trend
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h KAMA for adaptive trend
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        
        # Align 4h KAMA to 1h timeframe
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        
        # Calculate 4h trend direction (price vs KAMA)
        trend_4h = np.zeros(n)
        for i in range(n):
            if i < len(kama_4h_aligned) and kama_4h_aligned[i] > 0:
                if close[i] > kama_4h_aligned[i]:
                    trend_4h[i] = 1
                elif close[i] < kama_4h_aligned[i]:
                    trend_4h[i] = -1
    except Exception:
        kama_4h_aligned = np.zeros(n)
        trend_4h = np.zeros(n)
    
    # Get 1d data using mtf_data helper for regime filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        
        # 1d ADX for trend strength
        adx_1d = calculate_adx(h_1d, l_1d, c_1d, period=14)
        
        # Align 1d ADX to 1h timeframe
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    except Exception:
        adx_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Stochastic thresholds for entries
    STOCH_LONG_MAX = 30  # Oversold for long entry
    STOCH_SHORT_MIN = 70  # Overbought for short entry
    STOCH_EXIT = 50  # Neutral exit point
    
    # ADX minimum for regime filter (need trending market)
    ADX_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 14 * 2, 30)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(stoch_k_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get aligned MTF values
        trend_4h_val = trend_4h[i] if i < len(trend_4h) else 0
        adx_1d = adx_1d_aligned[i] if i < len(adx_1d_aligned) else 0
        
        # ADX regime filter - only trade when daily has trend strength
        if adx_1d < ADX_MIN:
            signals[i] = 0.0
            if position_side[i - 1] != 0 and i > 0:
                position_side[i] = 0  # Close position
            else:
                position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0 and i > 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Exit on stochastic cross back to neutral
            if prev_side == 1 and stoch_k_1h[i] > STOCH_EXIT:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            if prev_side == -1 and stoch_k_1h[i] < STOCH_EXIT:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 1h Stochastic pullback
        price = close[i]
        
        if trend_4h_val == 1:  # Bullish trend on 4h
            # Stochastic oversold on 1h (pullback entry)
            if stoch_k_1h[i] < STOCH_LONG_MAX and stoch_d_1h[i] < STOCH_LONG_MAX + 5:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h_val == -1:  # Bearish trend on 4h
            # Stochastic overbought on 1h (pullback entry)
            if stoch_k_1h[i] > STOCH_SHORT_MIN and stoch_d_1h[i] > STOCH_SHORT_MIN - 5:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals