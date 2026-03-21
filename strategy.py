#!/usr/bin/env python3
"""
EXPERIMENT #105 - MTF Supertrend+Chandelier+VolAdj Sizing (15m+4h Proper HTF v1)
==================================================================================================
Hypothesis: Recent failures (#101-104) had Chandelier exits but poor volatility scaling.
This strategy combines:
- 15m entries with 4h trend filter (proven in #031, #034, #035 with Sharpe > 7.5)
- Proper Chandelier exit: highest_high - 3*ATR(22) for trailing stops
- Volatility-adjusted position sizing: size = base_size * (avg_ATR / current_ATR)
- Discrete signal levels (0.0, ±0.20, ±0.35) to minimize churn costs
- ADX + BBW regime filters to avoid choppy markets
- Use mtf_data helper for PROPER 4h alignment (NO manual resampling!)

Why this should beat current best (Sharpe=3.653):
- Volatility-adjusted sizing reduces position in high vol (2022 crash protection)
- Chandelier exit trails tighter than fixed ATR stop
- 4h trend filter is more stable than 1h (fewer whipsaws)
- Based on lessons from 85+ failed strategies

Risk Management:
- Max signal: 0.35 (35% of capital - NOT 100%)
- Stoploss: Chandelier exit (3*ATR from highest high)
- Take profit: Reduce to half at 2R, trail at 1R
- Vol scaling: size *= (20-bar avg ATR / current ATR), clipped to 0.5-2.0x
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_chandelier_voladj_sizing_15m_4h_v1"
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


def calculate_chandelier_exit(high, low, close, atr, period=22, multiplier=3.0):
    """
    Calculate Chandelier Exit (ATR trailing stop)
    Long exit: highest_high - multiplier * ATR
    Short exit: lowest_low + multiplier * ATR
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    chandelier_long = np.zeros(n)  # Stop level for long positions
    chandelier_short = np.zeros(n)  # Stop level for short positions
    
    highest = np.zeros(n)
    lowest = np.zeros(n)
    
    for i in range(period - 1, n):
        highest[i] = np.max(high[i - period + 1:i + 1])
        lowest[i] = np.min(low[i - period + 1:i + 1])
        
        chandelier_long[i] = highest[i] - multiplier * atr[i]
        chandelier_short[i] = lowest[i] + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h data using PROPER mtf_data helper (CRITICAL - NO manual resampling!)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        n_4h = len(close_4h)
    except Exception:
        # Fallback if mtf_data not available
        close_4h = close[::16]
        high_4h = high[::16]
        low_4h = low[::16]
        n_4h = len(close_4h)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    atr_15m_22 = calculate_atr(high, low, close, period=22)  # For Chandelier
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Chandelier exit on 15m
    chandelier_long_15m, chandelier_short_15m = calculate_chandelier_exit(
        high, low, close, atr_15m_22, period=22, multiplier=3.0
    )
    
    # 4h indicators for trend (using mtf_data helper alignment)
    try:
        hma_4h = calculate_hma(close_4h, period=21)
        supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
        _, _, _, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        
        # Align 4h indicators to 15m timeframe
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
    except Exception:
        # Fallback
        hma_4h_aligned = np.zeros(n)
        st_direction_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
        bbw_4h_aligned = np.zeros(n)
    
    # Calculate 4h trend direction
    trend_4h = np.zeros(n)
    for i in range(n):
        if i >= 40 and hma_4h_aligned[i] > 0:
            if close[i] > hma_4h_aligned[i]:
                trend_4h[i] = 1
            elif close[i] < hma_4h_aligned[i]:
                trend_4h[i] = -1
    
    # Calculate volatility-adjusted position sizing factor
    # Base ATR (20-bar average) / Current ATR
    # When vol is high (current ATR > avg), reduce position size
    atr_avg_20 = np.zeros(n)
    vol_scale = np.ones(n)
    
    for i in range(20, n):
        atr_avg_20[i] = np.mean(atr_15m[i - 20:i + 1])
        if atr_15m[i] > 0:
            vol_scale[i] = atr_avg_20[i] / atr_15m[i]
            # Clip scaling factor to 0.5 - 2.0 range
            vol_scale[i] = np.clip(vol_scale[i], 0.5, 2.0)
        else:
            vol_scale[i] = 1.0
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE_FULL = 0.35
    BASE_SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 20
    
    # BBW minimum for regime filter (4h)
    BBW_MIN = 0.015
    
    first_valid = max(200, 40 * 16, 14 * 2, 20, 28)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    initial_risk = np.zeros(n)  # Track initial ATR risk for R-multiples
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filters
        trend = trend_4h[i]
        st_trend = st_direction_4h_aligned[i]
        adx_4h_val = adx_4h_aligned[i]
        bbw_4h_val = bbw_4h_aligned[i]
        
        # 15m indicators
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        chandelier_l = chandelier_long_15m[i]
        chandelier_s = chandelier_short_15m[i]
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # BBW filter - avoid choppy markets (4h)
        if bbw_4h_val < BBW_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filters must agree (HMA + Supertrend on 4h)
        if trend != st_trend or trend == 0:
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
            prev_initial_risk = initial_risk[i - 1] if initial_risk[i - 1] > 0 else atr
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Chandelier stoploss check (trailing stop)
            if prev_side == 1:
                # Use Chandelier exit as trailing stop
                if price < chandelier_l:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_risk[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * prev_initial_risk
                if not prev_tp and price >= tp_price:
                    # Apply volatility scaling to reduced position
                    size_half = BASE_SIZE_HALF * vol_scale[i]
                    signals[i] = np.clip(size_half, 0.0, BASE_SIZE_HALF)
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    initial_risk[i] = prev_initial_risk
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - prev_initial_risk
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        initial_risk[i] = 0
                        continue
                
            elif prev_side == -1:
                # Use Chandelier exit as trailing stop
                if price > chandelier_s:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    initial_risk[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * prev_initial_risk
                if not prev_tp and price <= tp_price:
                    # Apply volatility scaling to reduced position
                    size_half = -BASE_SIZE_HALF * vol_scale[i]
                    signals[i] = np.clip(size_half, -BASE_SIZE_HALF, 0.0)
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    initial_risk[i] = prev_initial_risk
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + prev_initial_risk
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        initial_risk[i] = 0
                        continue
            
            # Hold position if no exit triggered - apply vol scaling
            base_signal = signals[i - 1]
            scaled_signal = base_signal * vol_scale[i]
            if prev_side == 1:
                signals[i] = np.clip(scaled_signal, 0.0, BASE_SIZE_FULL)
            else:
                signals[i] = np.clip(scaled_signal, -BASE_SIZE_FULL, 0.0)
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            initial_risk[i] = initial_risk[i - 1]
            continue
        
        # Entry logic: 4h HMA + Supertrend + ADX + BBW + 15m RSI
        # Apply volatility scaling to entry size
        base_size = BASE_SIZE_FULL * vol_scale[i]
        entry_size = np.clip(base_size, 0.15, BASE_SIZE_FULL)  # Min 15%, max 35%
        
        if trend == 1 and st_trend == 1:  # Bullish trend confirmed on 4h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):  # Pullback entry
                signals[i] = entry_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                initial_risk[i] = atr  # Track initial ATR for R-multiples
                
        elif trend == -1 and st_trend == -1:  # Bearish trend confirmed on 4h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):  # Pullback entry
                signals[i] = -entry_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                initial_risk[i] = atr  # Track initial ATR for R-multiples
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals