#!/usr/bin/env python3
"""
EXPERIMENT #042 - MTF Donchian+ADX+RSI 1h+4h Simplified State
==================================================================================================
Hypothesis: Previous crashes (#040, #041) caused by numpy array mutation in loop.
Donchian breakouts + ADX trend strength filter should reduce whipsaw in choppy markets.
1h timeframe with 4h trend filter proven stable in #041 before crash.

Key changes from #041:
- Indicators: Donchian Channels (breakout), ADX (trend strength), RSI (momentum)
- State tracking: Pure Python lists, never modify numpy arrays in loop
- Signal levels: Discrete (0, ±0.25, ±0.30) to minimize churn costs
- ADX filter: Only trade when ADX > 25 (strong trend, avoid chop)
- Donchian: 20-period breakout with 4h confirmation
- Position sizing: Base 0.28 with ATR dynamic adjustment (0.22-0.32 range)

Why this should work:
- Donchian breakouts capture trending moves better than KAMA/EMA crossovers
- ADX > 25 filter avoids trading in sideways markets (major drawdown source)
- 4h trend + 1h breakout = proven MTF structure from best performer
- Simplified state tracking avoids numpy read-only crashes
- Conservative sizing (0.28 base) protects against 2022-style crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf


name = "mtf_donchian_adx_rsi_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return (
        np.nan_to_num(upper, nan=0.0),
        np.nan_to_num(lower, nan=0.0),
        np.nan_to_num(middle, nan=0.0)
    )


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        diff_high = high[i] - high[i-1]
        diff_low = low[i-1] - low[i]
        
        if diff_high > diff_low and diff_high > 0:
            plus_dm[i] = diff_high
        if diff_low > diff_high and diff_low > 0:
            minus_dm[i] = diff_low
    
    # Smooth with Wilder's method
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Initial values (first period)
    atr[period-1] = np.sum(tr[1:period]) / period
    plus_di[period-1] = 100 * np.sum(plus_dm[1:period]) / max(atr[period-1], 0.0001)
    minus_di[period-1] = 100 * np.sum(minus_dm[1:period]) / max(atr[period-1], 0.0001)
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_di[i] = (plus_di[i-1] * (period-1) + 100 * plus_dm[i] / max(atr[i], 0.0001)) / period
        minus_di[i] = (minus_di[i-1] * (period-1) + 100 * minus_dm[i] / max(atr[i], 0.0001)) / period
    
    # DX and ADX
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Initial ADX
    adx[period*2-1] = np.sum(dx[period:period*2]) / period
    
    # Smooth ADX
    for i in range(period*2, n):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return np.nan_to_num(adx, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    # Wilder's smoothing
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi = 100 - (100 / (1 + rs))
    return np.nan_to_num(rsi, nan=50.0)


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return np.nan_to_num(atr, nan=0.0)


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h HTF data using mtf_data helper (MANDATORY)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h trend indicators
    donchian_upper_4h, donchian_lower_4h, donchian_mid_4h = calculate_donchian(high_4h, low_4h, period=20)
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
    
    # Align 4h indicators to 1h timeframe (auto shift(1) for completed bars)
    donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1h entry indicators
    donchian_upper_1h, donchian_lower_1h, donchian_mid_1h = calculate_donchian(high, low, period=20)
    adx_1h = calculate_adx(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Parameters
    BASE_SIZE = 0.28
    MIN_SIZE = 0.22
    MAX_SIZE = 0.32
    ATR_STOP_MULT = 2.5
    TP_MULT = 2.0
    TRAIL_MULT = 1.0
    ADX_MIN = 25  # Only trade when trend is strong
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    signals = np.zeros(n)
    
    # Use Python lists for mutable state tracking (avoids numpy read-only issue)
    position_side = [0.0] * n
    entry_price = [0.0] * n
    tp_triggered = [0.0] * n
    extreme_price = [0.0] * n
    
    first_valid = max(100, 40 * 4)  # Ensure 4h data is aligned and indicators warmed up
    
    for i in range(first_valid, n):
        # Validate data
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 0:
            signals[i] = 0.0
            position_side[i] = 0.0
            entry_price[i] = 0.0
            tp_triggered[i] = 0.0
            extreme_price[i] = 0.0
            continue
        
        if np.isnan(donchian_mid_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            signals[i] = 0.0
            position_side[i] = 0.0
            entry_price[i] = 0.0
            tp_triggered[i] = 0.0
            extreme_price[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        
        # 4h trend direction (price vs Donchian middle)
        trend_4h = 0
        if price > donchian_mid_4h_aligned[i]:
            trend_4h = 1
        elif price < donchian_mid_4h_aligned[i]:
            trend_4h = -1
        
        # 4h ADX trend strength filter
        adx_strong = adx_4h_aligned[i] > ADX_MIN
        
        # Manage existing positions
        if position_side[i-1] != 0:
            prev_side = position_side[i-1]
            prev_entry = entry_price[i-1] if entry_price[i-1] > 0 else price
            prev_tp = tp_triggered[i-1]
            prev_extreme = extreme_price[i-1] if extreme_price[i-1] > 0 else prev_entry
            
            # Update extreme price
            if prev_side > 0:
                current_extreme = max(prev_extreme, price)
            else:
                current_extreme = min(prev_extreme, price) if prev_extreme > 0 else price
            extreme_price[i] = current_extreme
            
            # Stoploss check
            if prev_side > 0:
                stop_price = prev_entry - ATR_STOP_MULT * atr
                if price < stop_price:
                    signals[i] = 0.0
                    position_side[i] = 0.0
                    entry_price[i] = 0.0
                    tp_triggered[i] = 0.0
                    extreme_price[i] = 0.0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = BASE_SIZE * 0.5
                    position_side[i] = 1.0
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1.0
                    extreme_price[i] = current_extreme
                    continue
                
                # Trail stop at 1R after TP
                if prev_tp:
                    trail_price = current_extreme - TRAIL_MULT * ATR_STOP_MULT * atr
                    if price < trail_price:
                        signals[i] = 0.0
                        position_side[i] = 0.0
                        entry_price[i] = 0.0
                        tp_triggered[i] = 0.0
                        extreme_price[i] = 0.0
                        continue
            else:  # Short
                stop_price = prev_entry + ATR_STOP_MULT * atr
                if price > stop_price:
                    signals[i] = 0.0
                    position_side[i] = 0.0
                    entry_price[i] = 0.0
                    tp_triggered[i] = 0.0
                    extreme_price[i] = 0.0
                    continue
                
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -BASE_SIZE * 0.5
                    position_side[i] = -1.0
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1.0
                    extreme_price[i] = current_extreme
                    continue
                
                if prev_tp:
                    trail_price = current_extreme + TRAIL_MULT * ATR_STOP_MULT * atr
                    if price > trail_price:
                        signals[i] = 0.0
                        position_side[i] = 0.0
                        entry_price[i] = 0.0
                        tp_triggered[i] = 0.0
                        extreme_price[i] = 0.0
                        continue
            
            # Hold position - copy previous state
            signals[i] = signals[i-1]
            position_side[i] = position_side[i-1]
            entry_price[i] = entry_price[i-1]
            tp_triggered[i] = tp_triggered[i-1]
            extreme_price[i] = extreme_price[i-1]
            continue
        
        # Skip entries when ADX is weak (choppy market)
        if not adx_strong:
            signals[i] = 0.0
            position_side[i] = 0.0
            entry_price[i] = 0.0
            tp_triggered[i] = 0.0
            extreme_price[i] = 0.0
            continue
        
        # Entry logic: Donchian breakout + ADX strength + RSI confirmation
        rsi = rsi_1h[i]
        
        # 1h Donchian breakout signal
        donchian_breakout = 0
        if price > donchian_upper_1h[i]:
            donchian_breakout = 1
        elif price < donchian_lower_1h[i]:
            donchian_breakout = -1
        
        # Long entry: 4h uptrend + ADX strong + 1h Donchian breakout + RSI not overbought
        if trend_4h == 1 and adx_strong and donchian_breakout == 1:
            if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:
                # Dynamic sizing based on ATR
                atr_pct = atr / price
                vol_ratio = 0.015 / max(atr_pct, 0.001)
                vol_ratio = np.clip(vol_ratio, 0.7, 1.3)
                position_size = np.clip(BASE_SIZE * vol_ratio, MIN_SIZE, MAX_SIZE)
                
                signals[i] = position_size
                position_side[i] = 1.0
                entry_price[i] = price
                tp_triggered[i] = 0.0
                extreme_price[i] = price
                continue
        
        # Short entry: 4h downtrend + ADX strong + 1h Donchian breakout + RSI not oversold
        elif trend_4h == -1 and adx_strong and donchian_breakout == -1:
            if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:
                # Dynamic sizing based on ATR
                atr_pct = atr / price
                vol_ratio = 0.015 / max(atr_pct, 0.001)
                vol_ratio = np.clip(vol_ratio, 0.7, 1.3)
                position_size = np.clip(BASE_SIZE * vol_ratio, MIN_SIZE, MAX_SIZE)
                
                signals[i] = -position_size
                position_side[i] = -1.0
                entry_price[i] = price
                tp_triggered[i] = 0.0
                extreme_price[i] = price
                continue
        
        # No position
        signals[i] = 0.0
        position_side[i] = 0.0
        entry_price[i] = 0.0
        tp_triggered[i] = 0.0
        extreme_price[i] = 0.0
    
    return signals