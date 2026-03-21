#!/usr/bin/env python3
"""
EXPERIMENT #001 - MTF KAMA+Stochastic+ADX (30m+4h v1)
==================================================================================================
Hypothesis: 30m base timeframe with 4h KAMA trend filter + 30m Stochastic entry + ADX regime filter.
This differs from current best (mtf_hma_rsi_zscore_v1) by:
- KAMA instead of HMA (adapts to volatility, less whipsaw in chop)
- Stochastic instead of RSI (better for overbought/oversold in ranges)
- ADX filter instead of Z-score (filters low-trend periods explicitly)
- 30m timeframe instead of 1h (more signals, less noise than 15m)

Why this should work:
- KAMA's Efficiency Ratio adapts smoothing based on market regime
- Stochastic %K/%D cross gives clearer entry signals than RSI levels
- ADX > 25 ensures we only trade when trend has strength
- 30m proven sweet spot between signal frequency and noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_stoch_adx_30m_4h_v1"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
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
    
    pct_k = np.zeros(n)
    for i in range(n):
        if highest_high[i] > lowest_low[i]:
            pct_k[i] = 100 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
        else:
            pct_k[i] = 50
    
    pct_d = pd.Series(pct_k).rolling(window=d_period, min_periods=d_period).mean().values
    
    return pct_k, pct_d


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
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
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initial values
    sum_plus_dm = np.sum(plus_dm[1:period + 1])
    sum_minus_dm = np.sum(minus_dm[1:period + 1])
    sum_tr = np.sum(tr[1:period + 1])
    
    for i in range(period, n):
        if i == period:
            sum_plus_dm = np.sum(plus_dm[1:period + 1])
            sum_minus_dm = np.sum(minus_dm[1:period + 1])
            sum_tr = np.sum(tr[1:period + 1])
        else:
            sum_plus_dm = sum_plus_dm - plus_dm[i - 1] + plus_dm[i]
            sum_minus_dm = sum_minus_dm - minus_dm[i - 1] + minus_dm[i]
            sum_tr = sum_tr - tr[i - 1] + tr[i]
        
        if sum_tr > 0:
            plus_di[i] = 100 * sum_plus_dm / sum_tr
            minus_di[i] = 100 * sum_minus_dm / sum_tr
        else:
            plus_di[i] = 0
            minus_di[i] = 0
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # ADX is EMA of DX
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 30m indicators for entry timing
    atr_30m = calculate_atr(high, low, close, period=14)
    stoch_k_30m, stoch_d_30m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    adx_30m = calculate_adx(high, low, close, period=14)
    kama_30m = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
        atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
        
        # Align 4h indicators to 30m timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
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
        atr_4h_aligned = np.zeros(n)
        trend_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Stochastic thresholds for entry
    STCH_LONG_K_MIN = 20
    STCH_LONG_K_MAX = 50
    STCH_SHORT_K_MIN = 50
    STCH_SHORT_K_MAX = 80
    
    # ADX minimum for trend strength filter
    ADX_MIN = 22
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 14 * 3, 30 + 10)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    entry_atr = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_30m[i]) or np.isnan(stoch_k_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            if i > 0:
                position_side[i] = 0
            continue
        
        # Get aligned MTF values
        kama_4h_val = kama_4h_aligned[i] if i < len(kama_4h_aligned) else 0
        trend_4h_val = trend_4h[i] if i < len(trend_4h) else 0
        atr_4h_val = atr_4h_aligned[i] if i < len(atr_4h_aligned) else atr_30m[i]
        
        # ADX filter - only trade when trend has strength
        if adx_30m[i] < ADX_MIN:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                # Close position if ADX drops (trend weakening)
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
            else:
                position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            prev_atr = entry_atr[i - 1] if entry_atr[i - 1] > 0 else atr_30m[i - 1]
            
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
            
            # Stoploss check (2.0*ATR from entry)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * prev_atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * prev_atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    entry_atr[i] = prev_atr
                    continue
                
                # Trail stop at 1R profit after TP hit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * prev_atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        entry_atr[i] = 0
                        continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * prev_atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * prev_atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    entry_atr[i] = prev_atr
                    continue
                
                # Trail stop at 1R profit after TP hit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * prev_atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        entry_atr[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            entry_atr[i] = entry_atr[i - 1]
            continue
        
        # Entry logic: 4h trend + 30m Stochastic + ADX filter
        price = close[i]
        
        # Long entry: 4h bullish + Stochastic %K crossing up from oversold
        if trend_4h_val == 1 and kama_4h_val > 0:
            if (STCH_LONG_K_MIN <= stoch_k_30m[i] <= STCH_LONG_K_MAX and
                stoch_k_30m[i] > stoch_d_30m[i] and
                i > 0 and stoch_k_30m[i - 1] <= stoch_d_30m[i - 1]):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                entry_atr[i] = atr_30m[i]
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        # Short entry: 4h bearish + Stochastic %K crossing down from overbought
        elif trend_4h_val == -1 and kama_4h_val > 0:
            if (STCH_SHORT_K_MIN <= stoch_k_30m[i] <= STCH_SHORT_K_MAX and
                stoch_k_30m[i] < stoch_d_30m[i] and
                i > 0 and stoch_k_30m[i - 1] >= stoch_d_30m[i - 1]):
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                entry_atr[i] = atr_30m[i]
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals