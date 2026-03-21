#!/usr/bin/env python3
"""
EXPERIMENT #005 - MTF KAMA+Stoch+ADX (1h+4h+1d v1)
==================================================================================================
Hypothesis: Use 1h primary with 4h KAMA trend + 1h Stochastic entry + ADX strength filter + 1d regime.
This differs from current best by:
- KAMA (Kaufman Adaptive) instead of HMA - adapts to volatility automatically
- Stochastic oscillator instead of RSI - clearer overbought/oversold zones
- ADX filter for trend strength - only trade when trend is strong (>25)
- 1h primary timeframe - fewer signals than 15m, less churn costs
- Daily regime filter - avoid counter-trend trades on daily

Why this should work:
- KAMA ER (Efficiency Ratio) adjusts smoothing based on market noise
- Stochastic %K/%D crossover gives cleaner entry signals
- ADX > 25 ensures we're in trending market (not choppy)
- 1h has proven success in experiments, balances signal frequency vs noise
- Three timeframe confluence reduces false signals
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
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator %K and %D"""
    n = len(close)
    if n < k_period + d_period:
        return np.zeros(n), np.zeros(n)
    
    lowest_low = pd.Series(low).rolling(window=k_period, min_periods=k_period).min().values
    highest_high = pd.Series(high).rolling(window=k_period, min_periods=k_period).max().values
    
    k = np.zeros(n)
    for i in range(n):
        if highest_high[i] > lowest_low[i]:
            k[i] = 100 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
        else:
            k[i] = 50
    
    d = pd.Series(k).rolling(window=d_period, min_periods=d_period).mean().values
    
    return k, d


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    plus_sum = np.sum(plus_dm[1:period + 1])
    minus_sum = np.sum(minus_dm[1:period + 1])
    tr_sum = np.sum(tr[1:period + 1])
    
    for i in range(period, n):
        if i == period:
            plus_di[i] = 100 * plus_sum / tr_sum if tr_sum > 0 else 0
            minus_di[i] = 100 * minus_sum / tr_sum if tr_sum > 0 else 0
        else:
            plus_sum = plus_sum - plus_dm[i] + plus_dm[i + 1] if i + 1 < n else plus_sum - plus_dm[i]
            minus_sum = minus_sum - minus_dm[i] + minus_dm[i + 1] if i + 1 < n else minus_sum - minus_dm[i]
            tr_sum = tr_sum - tr[i] + tr[i + 1] if i + 1 < n else tr_sum - tr[i]
            
            plus_di[i] = 100 * plus_sum / tr_sum if tr_sum > 0 else 0
            minus_di[i] = 100 * minus_sum / tr_sum if tr_sum > 0 else 0
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    stoch_k_1h, stoch_d_1h = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    adx_1h = calculate_adx(high, low, close, period=14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
        
        # Align 4h indicators to 1h timeframe
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        c_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
    except Exception:
        kama_4h_aligned = np.zeros(n)
        c_4h_aligned = np.zeros(n)
    
    # Get 1d data using mtf_data helper for regime filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # Daily SMA for regime
        sma_1d = pd.Series(c_1d).rolling(window=50, min_periods=50).mean().values
        
        # Align 1d indicators to 1h timeframe
        sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
        c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    except Exception:
        sma_1d_aligned = np.zeros(n)
        c_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # Stochastic thresholds for entry
    STOC_LONG_K_MIN = 20
    STOC_LONG_K_MAX = 50
    STOC_SHORT_K_MIN = 50
    STOC_SHORT_K_MAX = 80
    
    # ADX minimum for trend strength
    ADX_MIN = 25
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 2, 30, 50)
    
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
        kama_4h_val = kama_4h_aligned[i] if i < len(kama_4h_aligned) else 0
        c_4h_val = c_4h_aligned[i] if i < len(c_4h_aligned) else 0
        sma_1d_val = sma_1d_aligned[i] if i < len(sma_1d_aligned) else 0
        c_1d_val = c_1d_aligned[i] if i < len(c_1d_aligned) else 0
        
        # 4h trend filter (price vs KAMA)
        trend_4h = 0
        if c_4h_val > 0 and kama_4h_val > 0:
            if c_4h_val > kama_4h_val:
                trend_4h = 1
            elif c_4h_val < kama_4h_val:
                trend_4h = -1
        
        # 1d regime filter (price vs SMA50)
        regime_1d = 0
        if c_1d_val > 0 and sma_1d_val > 0:
            if c_1d_val > sma_1d_val:
                regime_1d = 1
            elif c_1d_val < sma_1d_val:
                regime_1d = -1
        
        # ADX filter - only trade in trending markets
        if adx_1h[i] < ADX_MIN:
            signals[i] = 0.0
            if position_side[i - 1] != 0:
                position_side[i] = 0
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 1d regime + 1h Stochastic + ADX
        price = close[i]
        
        # Long entry: 4h bullish + 1d bullish + Stochastic crossover long + ADX strong
        if trend_4h == 1 and regime_1d == 1 and adx_1h[i] >= ADX_MIN:
            # Stochastic %K crosses above %D from oversold
            if (STOC_LONG_K_MIN <= stoch_k_1h[i] <= STOC_LONG_K_MAX and
                stoch_k_1h[i] > stoch_d_1h[i] and
                i > 0 and stoch_k_1h[i - 1] <= stoch_d_1h[i - 1]):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        # Short entry: 4h bearish + 1d bearish + Stochastic crossover short + ADX strong
        elif trend_4h == -1 and regime_1d == -1 and adx_1h[i] >= ADX_MIN:
            # Stochastic %K crosses below %D from overbought
            if (STOC_SHORT_K_MIN <= stoch_k_1h[i] <= STOC_SHORT_K_MAX and
                stoch_k_1h[i] < stoch_d_1h[i] and
                i > 0 and stoch_k_1h[i - 1] >= stoch_d_1h[i - 1]):
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