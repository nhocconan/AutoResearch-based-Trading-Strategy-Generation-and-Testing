#!/usr/bin/env python3
"""
EXPERIMENT #106 - MTF Supertrend+RSI+Chandelier+VolAdj Sizing (15m+4h Proper HTF v1)
==================================================================================================
Hypothesis: Recent failures (#101-#105) caused by improper MTF alignment and aggressive sizing.

Key changes from #040:
- Use mtf_data helper (get_htf_data, align_htf_to_ltf) for PROPER 4h alignment
- Chandelier exit: highest_high - 3*ATR(22) for trailing stop
- Volatility-adjusted sizing: base_size * (target_vol / current_vol)
- Discrete signal levels: 0.0, ±0.20, ±0.35 to reduce churning costs
- 4h trend filter + 15m entries (proven in #031, #034, #035)
- Position size MAX: 0.35 (critical for drawdown control per lessons learned)
- Stoploss: Chandelier exit (3*ATR) + initial 2*ATR stop

Why this should beat #040 and recent failures:
- Proper HTF alignment avoids data gap issues that killed #095-#097
- Volatility-adjusted sizing reduces exposure in high-vol regimes (2022 crash)
- Chandelier exit locks in profits during trends better than fixed ATR stops
- Based on #098 which had positive returns (Sharpe=0.145, Return=+80.7%)
- Simpler logic than #039-#040 complex MTF that may have overfitting
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
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rsi[i] = 100 - (100 / (1 + avg_gain[i] / avg_loss[i]))
    
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
    """Calculate Chandelier Exit (trailing stop)"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    chandelier_long = np.zeros(n)
    chandelier_short = np.zeros(n)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        chandelier_long[i] = highest_high - multiplier * atr[i]
        chandelier_short[i] = lowest_low + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def calculate_volatility(close, period=20):
    """Calculate rolling volatility (std of returns)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    returns = np.diff(close, prepend=close[0]) / np.where(close != 0, close, 1)
    vol = np.zeros(n)
    
    for i in range(period - 1, n):
        vol[i] = np.std(returns[i - period + 1:i + 1])
    
    return vol


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h data using mtf_data helper (CRITICAL for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        rsi_4h = calculate_rsi(close_4h, period=14)
        supertrend_4h, st_direction_4h = calculate_supertrend(
            high_4h, low_4h, close_4h, period=10, multiplier=3.0
        )
        
        # Align 4h indicators to 15m timeframe (auto shift(1) for completed bars)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        
        has_htf = True
    except Exception:
        # Fallback if mtf_data not available
        has_htf = False
        atr_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
        st_direction_4h_aligned = np.zeros(n)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    supertrend_15m, st_direction_15m = calculate_supertrend(
        high, low, close, period=10, multiplier=3.0
    )
    
    # Chandelier exit for trailing stop (22 period, 3*ATR)
    chandelier_long, chandelier_short = calculate_chandelier_exit(
        high, low, close, atr_15m, period=22, multiplier=3.0
    )
    
    # Volatility for position sizing adjustment
    vol_15m = calculate_volatility(close, period=20)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.35  # Max 35% of capital
    SIZE_HALF = 0.175
    TARGET_VOL = 0.015  # 1.5% daily vol target
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier (initial)
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 22, 14 * 2, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filter (if available)
        if has_htf:
            trend_4h = st_direction_4h_aligned[i]
            rsi_4h_val = rsi_4h_aligned[i]
            atr_4h_val = atr_4h_aligned[i]
        else:
            trend_4h = st_direction_15m[i]
            rsi_4h_val = rsi_15m[i]
            atr_4h_val = atr_15m[i]
        
        # 15m signals
        st_15m = st_direction_15m[i]
        rsi_15m_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        vol = vol_15m[i]
        
        # Volatility-adjusted position sizing
        if vol > 0:
            vol_adjustment = min(1.5, TARGET_VOL / vol)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = min(0.35, max(0.20, current_size))  # Clamp between 20-35%
        current_half = current_size / 2
        
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
            
            # Chandelier exit check (trailing stop)
            if prev_side == 1:
                chandelier_stop = chandelier_long[i]
                if price < chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Initial stoploss (2*ATR from entry)
                initial_stop = prev_entry - ATR_STOP_MULT * atr
                if price < initial_stop and not prev_tp:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = current_half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit after TP
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
                chandelier_stop = chandelier_short[i]
                if price > chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Initial stoploss (2*ATR from entry)
                initial_stop = prev_entry + ATR_STOP_MULT * atr
                if price > initial_stop and not prev_tp:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -current_half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit after TP
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 15m RSI pullback + Supertrend confirmation
        if trend_4h == 1 and st_15m == 1:  # Bullish trend confirmed
            if (RSI_LONG_MIN <= rsi_15m_val <= RSI_LONG_MAX):  # Pullback entry
                signals[i] = current_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1 and st_15m == -1:  # Bearish trend confirmed
            if (RSI_SHORT_MIN <= rsi_15m_val <= RSI_SHORT_MAX):  # Pullback entry
                signals[i] = -current_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals