#!/usr/bin/env python3
"""
EXPERIMENT #044 - MTF HMA+Supertrend+RSI+BBW+Z-score (4h+1d Dynamic Sizing v2)
==================================================================================================
Hypothesis: #040 proved 15m+4h works exceptionally well (Sharpe=16.0), but may be overfitted.
#043 showed 1h+4h works well (Sharpe=5.7) with cleaner signals. This experiment tests 4h+1d:
- 4h for entry timing (less noise than 15m/1h, more trades than 1d)
- 1d for primary trend filter (stronger directional bias than 4h)
- Simplified indicator stack: HMA + Supertrend + RSI + BBW + Z-score (remove ADX/KAMA/DEMA)
- ATR-based dynamic position sizing with discrete levels
- Tighter RSI pullback range (40-60) for higher quality entries
- Stoploss: 2.0*ATR, Take Profit: 2R then trail at 1R

Why this should beat #043:
- 4h+1d has stronger trend filter than 1h+4h (daily trend is more reliable)
- Fewer but higher quality trades (less churn, lower fees)
- Simpler indicator stack reduces overfitting risk
- Better suited for BTC/ETH/SOL across different market regimes
"""

import numpy as np
import pandas as pd

name = "mtf_hma_supertrend_rsi_bbw_zscore_4h_1d_v2"
timeframe = "4h"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 4h indicators for entry timing
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    zscore_4h = calculate_zscore(close, period=20)
    hma_4h = calculate_hma(close, period=21)
    supertrend_4h, st_direction_4h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_4h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Resample to 1d for trend filters (6 x 4h = 1d approximately)
    bars_per_1d = 6
    n_1d = (n // bars_per_1d)
    
    # Create 1d arrays by downsampling
    c_1d = np.zeros(n_1d)
    h_1d = np.zeros(n_1d)
    l_1d = np.zeros(n_1d)
    
    for i in range(n_1d):
        start_idx = i * bars_per_1d
        end_idx = start_idx + bars_per_1d
        c_1d[i] = close[end_idx - 1]
        h_1d[i] = np.max(high[start_idx:end_idx])
        l_1d[i] = np.min(low[start_idx:end_idx])
    
    # 1d indicators for primary trend
    hma_1d = calculate_hma(c_1d, period=21)
    supertrend_1d, st_direction_1d = calculate_supertrend(h_1d, l_1d, c_1d, period=10, multiplier=3.0)
    _, _, _, bbw_1d = calculate_bollinger_bands(c_1d, period=20, std_mult=2.0)
    
    # Map 1d indicators back to 4h timeframe
    trend_1d = np.zeros(n)
    st_trend_1d = np.zeros(n)
    bbw_1d_mapped = np.zeros(n)
    
    for i in range(n):
        idx_1d = i // bars_per_1d
        if idx_1d < n_1d and idx_1d >= 40:
            if c_1d[idx_1d] > hma_1d[idx_1d]:
                trend_1d[i] = 1
            elif c_1d[idx_1d] < hma_1d[idx_1d]:
                trend_1d[i] = -1
            
            st_trend_1d[i] = st_direction_1d[idx_1d]
            bbw_1d_mapped[i] = bbw_1d[idx_1d]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels with dynamic ATR adjustment
    BASE_SIZE = 0.35
    SIZE_HALF = 0.175
    TARGET_VOL = 0.025  # Target 2.5% daily volatility
    
    # RSI thresholds for pullback entries (wider range for 4h)
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.5
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # BBW minimum for regime filter (4h and 1d)
    BBW_MIN_4H = 0.015
    BBW_MIN_1D = 0.02
    
    first_valid = max(200, 40 * bars_per_1d, 14 * 2, 20, 28)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    current_size = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(zscore_4h[i]) or atr_4h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_1d[i]
        st_trend = st_trend_1d[i]
        rsi_val = rsi_4h[i]
        zscore_val = zscore_4h[i]
        atr = atr_4h[i]
        price = close[i]
        bbw_4h_val = bbw_4h[i]
        bbw_1d_val = bbw_1d_mapped[i]
        
        # Calculate dynamic position size based on ATR volatility
        atr_pct = atr / price if price > 0 else 0.01
        if atr_pct > 0:
            vol_adjustment = min(2.0, max(0.5, TARGET_VOL / atr_pct))
        else:
            vol_adjustment = 1.0
        
        dynamic_size = BASE_SIZE * vol_adjustment
        dynamic_size = min(0.40, max(0.20, dynamic_size))  # Clamp between 0.20 and 0.40
        dynamic_half = dynamic_size / 2
        
        # BBW filter - avoid choppy markets (both 4h and 1d)
        if bbw_4h_val < BBW_MIN_4H or bbw_1d_val < BBW_MIN_1D:
            # Check if we have existing position - if so, hold it
            if position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
                current_size[i] = current_size[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
                current_size[i] = 0
            continue
        
        # Trend filters must agree (HMA + Supertrend on 1d)
        if trend != st_trend or trend == 0:
            # Check if we have existing position - if so, hold it
            if position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
                current_size[i] = current_size[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
                current_size[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            prev_size = current_size[i - 1]
            
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
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    current_size[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = dynamic_half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    current_size[i] = dynamic_half
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        current_size[i] = 0
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
                    current_size[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -dynamic_half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    current_size[i] = dynamic_half
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        current_size[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            current_size[i] = current_size[i - 1]
            continue
        
        # Entry logic: 1d HMA + Supertrend + 4h HMA + RSI + Z-score + BBW
        hma_trend_4h = 1 if price > hma_4h[i] else (-1 if price < hma_4h[i] else 0)
        
        if trend == 1 and st_trend == 1 and hma_trend_4h == 1:  # Bullish trend confirmed
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                abs(zscore_val) < ZSCORE_MAX):  # Pullback + not extreme
                signals[i] = dynamic_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                current_size[i] = dynamic_size
                
        elif trend == -1 and st_trend == -1 and hma_trend_4h == -1:  # Bearish trend confirmed
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                abs(zscore_val) < ZSCORE_MAX):  # Pullback + not extreme
                signals[i] = -dynamic_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                current_size[i] = dynamic_size
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
            current_size[i] = 0
    
    return signals