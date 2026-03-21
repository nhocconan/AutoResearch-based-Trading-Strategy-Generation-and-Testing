#!/usr/bin/env python3
"""
EXPERIMENT #002 - MTF Supertrend+MACD+RSI (1h+4h)
==================================================================================================
Hypothesis: Supertrend provides cleaner trend signals than HMA/KAMA combo.
MACD histogram crosses offer better momentum entry timing than Stochastic.
1h timeframe reduces noise vs 15m while maintaining trade frequency.
Key design:
- 4H Supertrend(10,3) for primary trend direction
- 1H MACD(12,26,9) histogram cross for entry timing
- 1H RSI(14) for pullback confirmation (45-55 range)
- ATR(14) stoploss at 2.0x distance
- Position size: 0.30 (conservative, discrete levels)
- Take profit: reduce to half at 2R, trail stop at 1R

Why this should work:
- Supertrend is proven trend follower with built-in volatility adjustment
- MACD histogram momentum confirms trend continuation
- 1h timeframe balances signal quality vs trade frequency
- Simpler filter stack reduces overfitting risk
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_macd_rsi_1h_4h_v1"
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


def calculate_supertrend(high, low, close, period=10, mult=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        
        upper_band = hl2 + mult * atr[i]
        lower_band = hl2 - mult * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = -1
        else:
            if direction[i - 1] == 1:
                if close[i] > supertrend[i - 1]:
                    supertrend[i] = max(lower_band, supertrend[i - 1])
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band
                    direction[i] = -1
            else:
                if close[i] < supertrend[i - 1]:
                    supertrend[i] = min(upper_band, supertrend[i - 1])
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band
                    direction[i] = 1
    
    return supertrend, direction


def calculate_ema(close, period=12):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    multiplier = 2.0 / (period + 1)
    
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    valid_start = slow + signal - 1
    
    signal_line[valid_start] = np.mean(macd_line[slow:valid_start + 1])
    
    multiplier = 2.0 / (signal + 1)
    for i in range(valid_start + 1, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def resample_to_higher_tf(prices, target_tf='4h'):
    """Resample to higher timeframe using open_time index - CRITICAL for no look-ahead"""
    prices_indexed = prices.set_index('open_time')
    df_resampled = prices_indexed.resample(target_tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return df_resampled


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    macd_line_1h, signal_line_1h, histogram_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Resample to 4h for trend filter using proper method
    try:
        df_4h = resample_to_higher_tf(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        n_4h = len(c_4h)
        
        # 4h Supertrend for trend direction
        _, supertrend_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, mult=3.0)
        
        # Map 4h indicators back to 1h timeframe
        prices_indexed = prices.set_index('open_time')
        df_4h_indexed = df_4h.set_index('open_time')
        
        # Align 4h data to 1h timestamps with shift(1) to avoid look-ahead
        trend_4h = np.zeros(n)
        
        # Create reindexed 4h trend with proper alignment
        trend_4h_series = pd.Series(supertrend_dir_4h, index=df_4h_indexed.index)
        # CRITICAL: shift by 1 to only use COMPLETED 4h bars
        trend_4h_shifted = trend_4h_series.shift(1)
        trend_aligned = trend_4h_shifted.reindex(prices_indexed.index, method='ffill').fillna(0).values
        
        trend_4h = trend_aligned
        
    except Exception:
        # Fallback if resampling fails
        bars_per_4h = 4  # 1h bars per 4h bar
        n_4h = n // bars_per_4h
        
        c_4h = np.zeros(n_4h)
        h_4h = np.zeros(n_4h)
        l_4h = np.zeros(n_4h)
        
        for i in range(n_4h):
            start_idx = i * bars_per_4h
            end_idx = start_idx + bars_per_4h
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
        
        _, supertrend_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, mult=3.0)
        
        trend_4h = np.zeros(n)
        for i in range(n):
            idx_4h = i // bars_per_4h
            if idx_4h > 0 and idx_4h < n_4h:
                # Use previous completed 4h bar (shift by 1)
                trend_4h[i] = supertrend_dir_4h[idx_4h - 1] if idx_4h > 0 else 0
    
    signals = np.zeros(n)
    
    # Position sizing - conservative discrete levels
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Entry thresholds
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 55
    
    # MACD histogram threshold for momentum confirmation
    MACD_MIN = 0.0
    
    # Stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Minimum bars for valid signals
    first_valid = max(200, 40 * 4, 26 + 9, 14 * 2)
    
    # Position tracking for stoploss and take profit
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        rsi_val = rsi_1h[i]
        macd_hist = histogram_1h[i]
        macd_hist_prev = histogram_1h[i - 1] if i > 0 else 0.0
        atr = atr_1h[i]
        price = close[i]
        
        # Check if we have an open position
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Track extremes since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Check stoploss
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Check take profit (2R)
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop after TP (1R from highest)
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Check take profit (2R)
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop after TP (1R from lowest)
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Maintain position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # No open position - check for new entry
        # Long entry: 4h Supertrend bullish + MACD histogram crossing up + RSI in pullback range
        if trend == 1:
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and
                macd_hist > MACD_MIN and
                macd_hist_prev <= MACD_MIN):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        # Short entry: 4h Supertrend bearish + MACD histogram crossing down + RSI in pullback range
        elif trend == -1:
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and
                macd_hist < -MACD_MIN and
                macd_hist_prev >= -MACD_MIN):
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
    
    return signals