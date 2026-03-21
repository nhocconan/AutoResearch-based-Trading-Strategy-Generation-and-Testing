#!/usr/bin/env python3
"""
EXPERIMENT #017 - MTF Donchian+MACD+ADX+Z-score (4h+1h v1)
==================================================================================================
Hypothesis: Current best #012 uses 1h+4h with EMA+Supertrend+MACD+RSI+ADX+Z-score (Sharpe=0.478).
This experiment tries a DIFFERENT combination:
- 4H Donchian channel for trend direction (breakout-based, different from HMA/EMA)
- 1H MACD histogram cross for entry timing (momentum-based, different from RSI pullback)
- ADX(14) > 25 for trend strength confirmation
- Z-score(20) < 2.0 to avoid extreme mean reversion scenarios
- Signal size: 0.30 (conservative, proven safe range)
- Stoploss: 2.0*ATR with trailing after 1R profit

Why this should beat #012:
- Donchian channels capture breakouts better than moving averages in trending markets
- MACD histogram crosses provide cleaner momentum signals than RSI pullbacks
- 4h trend + 1h entry is proven stable (from #005, #006, #012)
- Simpler MTF logic reduces bugs (learned from #037-#039 failures)
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_macd_adx_zscore_4h_1h_v1"
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (EMA fast - EMA slow, signal line, histogram)"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    first_signal = np.mean(macd_line[slow:slow + signal])
    signal_line[slow + signal - 1] = first_signal
    
    for i in range(slow + signal, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
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
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


def calculate_ema(close, period=14):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = ema[i - 1] + (2.0 / (period + 1)) * (close[i] - ema[i - 1])
    
    return ema


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    macd_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    ema_1h = calculate_ema(close, period=50)
    
    # Resample to 4h for trend filters using open_time
    prices_indexed = prices.set_index('open_time')
    
    # Resample to 4h
    df_4h = prices_indexed.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h indicators
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    n_4h = len(c_4h)
    
    donchian_upper_4h, donchian_middle_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
    ema_4h = calculate_ema(c_4h, period=50)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h indicators back to 1h using reindex with ffill
    df_4h['donchian_upper'] = donchian_upper_4h
    df_4h['donchian_middle'] = donchian_middle_4h
    df_4h['donchian_lower'] = donchian_lower_4h
    df_4h['ema_50'] = ema_4h
    df_4h['adx'] = adx_4h
    
    # Reindex to 1h timestamps with forward fill
    df_4h_aligned = df_4h.reindex(prices_indexed.index, method='ffill')
    
    donchian_upper_aligned = df_4h_aligned['donchian_upper'].values
    donchian_middle_aligned = df_4h_aligned['donchian_middle'].values
    donchian_lower_aligned = df_4h_aligned['donchian_lower'].values
    ema_50_4h_aligned = df_4h_aligned['ema_50'].values
    adx_4h_aligned = df_4h_aligned['adx'].values
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # MACD histogram thresholds for entry
    MACD_HIST_MIN = 0.0
    MACD_HIST_MAX = 0.0  # Cross through zero
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 25
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 50, 26 + 9, 28)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(macd_hist_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(ema_50_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h Donchian trend direction
        price = close[i]
        donchian_mid = donchian_middle_aligned[i]
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        ema_50_4h = ema_50_4h_aligned[i]
        adx_4h_val = adx_4h_aligned[i]
        
        # Determine 4h trend from Donchian position
        if price > donchian_mid and price > ema_50_4h:
            trend_4h = 1
        elif price < donchian_mid and price < ema_50_4h:
            trend_4h = -1
        else:
            trend_4h = 0
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Z-score filter - avoid extreme mean reversion scenarios
        zscore_val = zscore_1h[i]
        if abs(zscore_val) > ZSCORE_MAX:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
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
        
        # Entry logic: 4h Donchian + EMA trend + 1h MACD histogram cross + ADX + Z-score
        macd_hist = macd_hist_1h[i]
        macd_hist_prev = macd_hist_1h[i - 1] if i > 0 else 0
        
        if trend_4h == 1:  # Bullish trend on 4h
            # MACD histogram crossing above zero (momentum confirmation)
            if macd_hist > 0 and macd_hist_prev <= 0:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1:  # Bearish trend on 4h
            # MACD histogram crossing below zero (momentum confirmation)
            if macd_hist < 0 and macd_hist_prev >= 0:
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