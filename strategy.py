#!/usr/bin/env python3
"""
EXPERIMENT #030 - MTF Supertrend+BBW+RSI (15m+4h Simplified Robust v3)
==================================================================================================
Hypothesis: Previous crashes were due to complex position state tracking and array indexing.
This version uses simplified signal generation without complex state variables.
Key improvements:
- Remove complex position state tracking (in_position, entry_price, etc.)
- Use vectorized signal generation where possible
- Simpler stoploss logic based on price vs supertrend
- 15m entries + 4h trend (proven from #025 which had Sharpe=0.025)
- Position size: 0.30 (balanced for risk/return)
- Add Bollinger Band Width for regime filter (avoid low volatility chop)

Why this should work:
- Simpler logic = fewer crash opportunities
- 15m timeframe captures more opportunities than 1h
- BBW filter avoids choppy markets
- Supertrend provides clean trend direction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_bbw_rsi_15m_4h_v3"
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
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        elif avg_gain[i] == 0:
            rsi[i] = 0.0
        else:
            rsi[i] = 100 - (100 / (1 + avg_gain[i] / avg_loss[i]))
    
    return np.nan_to_num(rsi, nan=50.0)


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.zeros(n)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    # Initialize
    supertrend[period] = lower_band[period]
    trend_direction[period] = 1
    
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
    """Calculate Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # Bandwidth = (upper - lower) / sma
    bw = np.zeros(n)
    for i in range(period - 1, n):
        if sma[i] > 0:
            bw[i] = (upper[i] - lower[i]) / sma[i]
        else:
            bw[i] = 0.0
    
    return upper, lower, bw


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if volatility == 0:
            er = 1.0
        else:
            er = change / volatility
        
        sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Initialize signals array
    signals = np.zeros(n)
    
    # Calculate 15m indicators
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # 4h trend filters using mtf_data helper (MANDATORY)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h indicators on actual 4h data
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    kama_4h = calculate_kama(close_4h, period=10)
    supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    
    # Align 4h indicators to 15m timeframe (auto shift for completed bars only)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    SIZE_ZERO = 0.0
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 50
    RSI_SHORT_MIN = 50
    RSI_SHORT_MAX = 65
    
    # BBW percentile for regime filter (avoid chop)
    bbw_window = 100
    bbw_percentile_low = 30  # Avoid very low volatility
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Minimum bars for valid signals
    first_valid = max(100, int(n * 0.05))
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0:
            signals[i] = SIZE_ZERO
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_4h_aligned[i]) or np.isnan(st_direction_4h_aligned[i]):
            signals[i] = SIZE_ZERO
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # 4h trend direction
        trend_4h = 0
        if kama_4h_aligned[i] > 0 and close[i] > kama_4h_aligned[i]:
            trend_4h = 1
        elif kama_4h_aligned[i] > 0 and close[i] < kama_4h_aligned[i]:
            trend_4h = -1
        
        st_trend_4h = int(st_direction_4h_aligned[i])
        
        # 15m indicators
        rsi_val = rsi_15m[i]
        st_direction_15m = int(st_direction_15m[i])
        atr = atr_15m[i]
        price = close[i]
        bbw = bbw_15m[i]
        
        # Calculate BBW percentile (rolling)
        bbw_percentile = 50.0
        if i >= bbw_window:
            bbw_history = bbw_15m[i - bbw_window:i + 1]
            bbw_percentile = np.percentile(bbw_history, 50)  # Current BBW vs history
        
        # Exit logic for existing positions
        if in_position:
            # Update highest/lowest since entry
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                stoploss_price = entry_price - ATR_STOP_MULT * entry_atr
                
                # Stoploss check
                if price < stoploss_price:
                    signals[i] = SIZE_ZERO
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = entry_price + 2 * ATR_STOP_MULT * entry_atr
                if not tp_triggered and price >= tp_price:
                    signals[i] = SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if tp_triggered:
                    trail_stop = highest_since_entry - ATR_STOP_MULT * entry_atr
                    if price < trail_stop:
                        signals[i] = SIZE_ZERO
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
                
            elif position_side == -1:
                lowest_since_entry = min(lowest_since_entry, price)
                stoploss_price = entry_price + ATR_STOP_MULT * entry_atr
                
                # Stoploss check
                if price > stoploss_price:
                    signals[i] = SIZE_ZERO
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = entry_price - 2 * ATR_STOP_MULT * entry_atr
                if not tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if tp_triggered:
                    trail_stop = lowest_since_entry + ATR_STOP_MULT * entry_atr
                    if price > trail_stop:
                        signals[i] = SIZE_ZERO
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1] if i > 0 else SIZE_ZERO
            continue
        
        # BBW regime filter (avoid very low volatility chop)
        bbw_hist = bbw_15m[max(0, i - bbw_window):i + 1]
        if len(bbw_hist) >= 10:
            bbw_percentile = np.percentile(bbw_hist, 50)
            if bbw < bbw_percentile * 0.7:  # BBW too low = avoid
                signals[i] = SIZE_ZERO
                continue
        
        # Entry logic: 4h trend + 15m RSI pullback + 15m Supertrend confirmation
        if trend_4h == 1 and st_trend_4h == 1:  # Bullish trend on 4h
            # Wait for RSI pullback on 15m, confirm with 15m Supertrend
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                st_direction_15m == 1):
                signals[i] = SIZE_FULL
                in_position = True
                position_side = 1
                entry_price = price
                entry_atr = atr
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                
        elif trend_4h == -1 and st_trend_4h == -1:  # Bearish trend on 4h
            # Wait for RSI pullback on 15m, confirm with 15m Supertrend
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                st_direction_15m == -1):
                signals[i] = -SIZE_FULL
                in_position = True
                position_side = -1
                entry_price = price
                entry_atr = atr
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
        
        else:
            signals[i] = SIZE_ZERO
    
    return signals