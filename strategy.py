#!/usr/bin/env python3
"""
EXPERIMENT #005 - MTF EMA+ADX+Stoch (15m+1h+4h v1)
==================================================================================================
Hypothesis: Combine 4h EMA crossover (21/55) for clear trend direction + 1h ADX (14) for trend 
strength filter + 15m Stochastic (14,3,3) for oversold/overbought entry timing. This differs from 
current best by:
- EMA crossover instead of HMA for trend (proven classic trend filter)
- ADX strength filter to avoid weak trends (vs Z-score)
- Stochastic for entry timing (vs RSI)
- Three timeframes: 15m base, 1h strength, 4h trend

Why this should work:
- 4h EMA(21/55) crossover is a proven trend filter with clear signals
- ADX > 25 ensures we only trade strong trends (reduces whipsaws in chop)
- Stochastic gives clear overbought/oversold levels for pullback entries
- 15m base timeframe has proven success in previous experiments
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_ema_adx_stoch_15m_1h_4h_v1"
timeframe = "15m"
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


def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = pd.Series(close).ewm(span=period, adjust=False).mean().values
    return ema


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > 0 and high_diff > low_diff:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
        
        if low_diff > 0 and low_diff > high_diff:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    adx = np.zeros(n)
    
    plus_smooth = np.zeros(n)
    minus_smooth = np.zeros(n)
    tr_smooth = np.zeros(n)
    
    plus_smooth[period - 1] = np.sum(plus_dm[:period])
    minus_smooth[period - 1] = np.sum(minus_dm[:period])
    tr_smooth[period - 1] = np.sum(tr[:period])
    
    for i in range(period, n):
        plus_smooth[i] = plus_smooth[i - 1] - plus_smooth[i - 1] / period + plus_dm[i]
        minus_smooth[i] = minus_smooth[i - 1] - minus_smooth[i - 1] / period + minus_dm[i]
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / period + tr[i]
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_smooth[i] / tr_smooth[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    dx = np.zeros(n)
    for i in range(period, n):
        if (plus_di[i] + minus_di[i]) > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator (%K and %D)"""
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    stoch_k_15m, stoch_d_15m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    
    # Get 1h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        h_1h = df_1h['high'].values
        l_1h = df_1h['low'].values
        
        # 1h ADX for trend strength
        adx_1h = calculate_adx(h_1h, l_1h, c_1h, period=14)
        
        # Align 1h indicators to 15m timeframe (auto shift for completed bars)
        adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    except Exception:
        # Fallback if mtf_data fails
        adx_1h_aligned = np.zeros(n)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        
        # 4h EMA crossover for trend direction
        ema21_4h = calculate_ema(c_4h, period=21)
        ema55_4h = calculate_ema(c_4h, period=55)
        
        # Align 4h indicators to 15m timeframe
        ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
        ema55_4h_aligned = align_htf_to_ltf(prices, df_4h, ema55_4h)
        
        # Calculate 4h trend direction (EMA21 vs EMA55)
        trend_4h = np.zeros(n)
        for i in range(n):
            if i < len(ema21_4h_aligned) and i < len(ema55_4h_aligned):
                if ema21_4h_aligned[i] > ema55_4h_aligned[i]:
                    trend_4h[i] = 1
                elif ema21_4h_aligned[i] < ema55_4h_aligned[i]:
                    trend_4h[i] = -1
    except Exception:
        trend_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Stochastic thresholds for entry
    STOCH_LONG_MAX = 30  # Enter long when stochastic is oversold
    STOCH_SHORT_MIN = 70  # Enter short when stochastic is overbought
    
    # ADX threshold for trend strength
    ADX_MIN = 25
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 2, 55, 14 + 3)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(stoch_k_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        adx_1h = adx_1h_aligned[i] if i < len(adx_1h_aligned) else 0
        trend_4h_val = trend_4h[i] if i < len(trend_4h) else 0
        
        # ADX filter - only trade when trend is strong enough (1h)
        if adx_1h < ADX_MIN:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                # Close position if trend weakens
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                position_side[i] = 0
            continue
        
        # 4h trend filter (EMA crossover must show direction)
        if trend_4h_val == 0:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                # Close position if trend unclear
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
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
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
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
        
        # Entry logic: 4h trend + 1h ADX strength + 15m Stochastic pullback
        price = close[i]
        
        if trend_4h_val == 1:  # Bullish trend on 4h (EMA21 > EMA55)
            # Stochastic oversold on 15m (pullback entry)
            if stoch_k_15m[i] <= STOCH_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h_val == -1:  # Bearish trend on 4h (EMA21 < EMA55)
            # Stochastic overbought on 15m (pullback entry)
            if stoch_k_15m[i] >= STOCH_SHORT_MIN:
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