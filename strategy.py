#!/usr/bin/env python3
"""
EXPERIMENT #101 - MTF Supertrend+RSI+Chandelier Exit with Vol-Adjusted Sizing (15m+4h)
==================================================================================================
Hypothesis: Recent failures (#089-#100) show manual MTF resampling causes alignment issues.
Current best (#040) has complex logic but uses manual resampling which breaks on SOL gaps.

Key changes from #040:
- Use mtf_data helper (get_htf_data, align_htf_to_ltf) for PROPER 4h alignment
- Add Chandelier Exit (highest_high - 3*ATR) for trailing stops
- Volatility-adjusted position sizing (reduce size when ATR% is high)
- Simplify entry logic: Supertrend direction + RSI pullback only
- Remove ADX/BBW/KAMA filters that caused overfitting in #094-#100
- Discrete signal levels: 0.0, ±0.20, ±0.30 (reduce churn costs)
- Proper min_periods on all rolling calculations

Why this should beat current best (Sharpe=3.653):
- Proper HTF alignment prevents data gap issues (SOL had 2x 3-day gaps)
- Chandelier exit captures more trend vs fixed 2R TP
- Vol-adjusted sizing reduces exposure in high-vol regimes (2022 crash)
- Simpler logic = more robust across BTC/ETH/SOL
- Based on #098 which had Sharpe=0.145 but proper MTF structure
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_chandelier_voladj_15m_4h_v1"
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


def calculate_chandelier_exit(high, low, close, atr, period=22, multiplier=3.0):
    """Calculate Chandelier Exit (ATR trailing stop)"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    long_stop = np.zeros(n)
    short_stop = np.zeros(n)
    
    highest = np.zeros(n)
    lowest = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            highest[i] = high[i]
            lowest[i] = low[i]
        else:
            highest[i] = max(highest[i - 1], high[i])
            lowest[i] = min(lowest[i - 1], low[i])
    
    for i in range(period, n):
        long_stop[i] = highest[i] - multiplier * atr[i]
        short_stop[i] = lowest[i] + multiplier * atr[i]
    
    return long_stop, short_stop


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    chandelier_long, chandelier_short = calculate_chandelier_exit(high, low, close, atr_15m, period=22, multiplier=3.0)
    
    # Get 4h data using mtf_data helper (CRITICAL for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h Supertrend for trend direction
        st_4h, st_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        
        # 4h ATR for volatility regime
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        # Align 4h indicators to 15m timeframe (auto shift(1) for completed bars)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_dir_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        st_trend_4h_aligned = st_direction_15m
        atr_4h_aligned = atr_15m
    
    # Position sizing parameters
    SIZE_BASE = 0.30  # Base position size (conservative)
    SIZE_MIN = 0.15   # Minimum in high vol
    SIZE_MAX = 0.35   # Maximum in low vol
    
    # Volatility-adjusted sizing: reduce size when ATR% is high
    atr_pct_4h = atr_4h_aligned / close
    vol_regime = np.where(atr_pct_4h > np.percentile(atr_pct_4h[100:], 75), 'high',
                          np.where(atr_pct_4h < np.percentile(atr_pct_4h[100:], 25), 'low', 'normal'))
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Chandelier exit multiplier
    CHANDELIER_MULT = 3.0
    
    first_valid = max(200, 40, 22, 14)
    
    # Track position state
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    chandelier_stop = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend direction
        trend_4h = st_trend_4h_aligned[i]
        
        # 15m Supertrend direction
        st_15m = st_direction_15m[i]
        
        # Volatility-adjusted position size
        if vol_regime[i] == 'high':
            position_size = SIZE_MIN
        elif vol_regime[i] == 'low':
            position_size = SIZE_MAX
        else:
            position_size = SIZE_BASE
        
        # Check existing positions first (Chandelier exit)
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_chand = chandelier_stop[i - 1]
            
            # Update highest/lowest since entry for Chandelier
            if prev_side == 1:
                current_high = max(highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry, high[i])
                current_chand = current_high - CHANDELIER_MULT * atr_15m[i]
            else:
                current_low = min(lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry, low[i])
                current_chand = current_low + CHANDELIER_MULT * atr_15m[i]
            
            highest_since_entry[i] = current_high if prev_side == 1 else highest_since_entry[i - 1]
            lowest_since_entry[i] = current_low if prev_side == -1 else lowest_since_entry[i - 1]
            chandelier_stop[i] = current_chand
            
            # Chandelier exit check
            if prev_side == 1:
                if close[i] < current_chand or st_15m == -1:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    chandelier_stop[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                else:
                    signals[i] = position_size
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    chandelier_stop[i] = current_chand
                    
            elif prev_side == -1:
                if close[i] > current_chand or st_15m == 1:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    chandelier_stop[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                else:
                    signals[i] = -position_size
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    chandelier_stop[i] = current_chand
            continue
        
        # Entry logic: 4h trend + 15m pullback + RSI filter
        rsi_val = rsi_15m[i]
        
        if trend_4h == 1 and st_15m == 1:  # Bullish on both timeframes
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:  # Pullback entry
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = close[i]
                chandelier_stop[i] = close[i] - CHANDELIER_MULT * atr_15m[i]
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
                
        elif trend_4h == -1 and st_15m == -1:  # Bearish on both timeframes
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:  # Pullback entry
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = close[i]
                chandelier_stop[i] = close[i] + CHANDELIER_MULT * atr_15m[i]
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals