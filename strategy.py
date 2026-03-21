#!/usr/bin/env python3
"""
EXPERIMENT #007 - MTF KAMA+Stoch+ADX (30m+4h+1d v1)
==================================================================================================
Hypothesis: KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA.
Combined with Stochastic oscillator for entry timing and ADX for trend strength filter.
Primary=30m (different from failed 15m/1h attempts), HTF=4h trend + 1d regime.

Why this should work:
- KAMA adjusts smoothing based on market noise (ER ratio)
- Stochastic provides clearer overbought/oversold than RSI
- ADX > 25 filters out choppy/weak trend periods
- 30m timeframe balances noise vs opportunity frequency
- Three timeframe confluence: 1d regime + 4h trend + 30m entry
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_stoch_adx_30m_4h_1d_v1"
timeframe = "30m"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = change / noise
        else:
            er[i] = 0
    
    # Smoothing constant
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator (%K and %D)"""
    n = len(close)
    if n < k_period + d_period:
        return np.zeros(n), np.zeros(n)
    
    k = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest = np.min(low[i - k_period + 1:i + 1])
        highest = np.max(high[i - k_period + 1:i + 1])
        if highest - lowest > 0:
            k[i] = 100 * (close[i] - lowest) / (highest - lowest)
        else:
            k[i] = 50
    
    # %D is SMA of %K
    for i in range(k_period - 1 + d_period - 1, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    n = len(close)
    if n < period * 3:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # First TR sum
    tr_sum = np.sum(tr[1:period + 1])
    plus_dm_sum = np.sum(plus_dm[1:period + 1])
    minus_dm_sum = np.sum(minus_dm[1:period + 1])
    
    for i in range(period, n):
        if i == period:
            plus_di[i] = 100 * plus_dm_sum / tr_sum if tr_sum > 0 else 0
            minus_di[i] = 100 * minus_dm_sum / tr_sum if tr_sum > 0 else 0
        else:
            plus_di[i] = 100 * plus_dm[i] / tr[i] if tr[i] > 0 else 0
            minus_di[i] = 100 * minus_dm[i] / tr[i] if tr[i] > 0 else 0
        
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    # ADX is SMA of DX
    for i in range(period * 2 - 1, n):
        adx[i] = np.mean(dx[i - period + 1:i + 1])
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 30m indicators for entry timing
    atr_30m = calculate_atr(high, low, close, period=14)
    kama_30m = calculate_kama(close, period=10, fast=2, slow=30)
    stoch_k_30m, stoch_d_30m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    adx_30m = calculate_adx(high, low, close, period=14)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h indicators
        kama_4h = calculate_kama(c_4h, period=10, fast=2, slow=30)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        # Align 4h indicators to 30m timeframe
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    except Exception:
        kama_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
    
    # Get 1d data using mtf_data helper for regime filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # 1d trend (price vs KAMA)
        kama_1d = calculate_kama(c_1d, period=10, fast=2, slow=30)
        trend_1d = np.where(c_1d > kama_1d, 1, np.where(c_1d < kama_1d, -1, 0))
        
        # Align 1d trend to 30m timeframe
        trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    except Exception:
        trend_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Stochastic thresholds
    STOCH_LONG_ENTRY = 25  # Oversold for long entry
    STOCH_SHORT_ENTRY = 75  # Overbought for short entry
    STOCH_EXIT = 50  # Neutral exit
    
    # ADX threshold for trend strength
    ADX_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 14 * 3, 30 + 10)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_30m[i]) or np.isnan(kama_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            if i > 0:
                position_side[i] = position_side[i - 1]
            continue
        
        # Get aligned MTF values
        kama_4h = kama_4h_aligned[i] if i < len(kama_4h_aligned) else close[i]
        adx_4h = adx_4h_aligned[i] if i < len(adx_4h_aligned) else 0
        trend_1d = trend_1d_aligned[i] if i < len(trend_1d_aligned) else 0
        
        # 4h trend direction
        trend_4h = 1 if close[i] > kama_4h else -1 if close[i] < kama_4h else 0
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
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
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_30m[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_30m[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_30m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Exit on Stochastic overbought
                if stoch_k_30m[i] > STOCH_EXIT and stoch_d_30m[i] > STOCH_EXIT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_30m[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_30m[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_30m[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Exit on Stochastic oversold
                if stoch_k_30m[i] < STOCH_EXIT and stoch_d_30m[i] < STOCH_EXIT:
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
        
        # Entry logic: 1d regime + 4h trend + 30m Stochastic entry + ADX filter
        price = close[i]
        
        # ADX filter - only trade when trend is strong enough
        if adx_4h < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 1d regime filter
        if trend_1d == 1:  # Bullish daily regime
            # 4h trend must agree
            if trend_4h == 1:
                # 30m Stochastic oversold for long entry
                if stoch_k_30m[i] < STOCH_LONG_ENTRY and stoch_d_30m[i] < STOCH_LONG_ENTRY + 10:
                    # Price above 4h KAMA confirmation
                    if price > kama_4h:
                        signals[i] = SIZE_FULL
                        position_side[i] = 1
                        entry_price[i] = price
                        tp_triggered[i] = 0
                        highest_since_entry[i] = price
                        lowest_since_entry[i] = price
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend_1d == -1:  # Bearish daily regime
            # 4h trend must agree
            if trend_4h == -1:
                # 30m Stochastic overbought for short entry
                if stoch_k_30m[i] > STOCH_SHORT_ENTRY and stoch_d_30m[i] > STOCH_SHORT_ENTRY - 10:
                    # Price below 4h KAMA confirmation
                    if price < kama_4h:
                        signals[i] = -SIZE_FULL
                        position_side[i] = -1
                        entry_price[i] = price
                        tp_triggered[i] = 0
                        highest_since_entry[i] = price
                        lowest_since_entry[i] = price
                    else:
                        signals[i] = 0.0
                        position_side[i] = 0
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals