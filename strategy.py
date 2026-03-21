#!/usr/bin/env python3
"""
EXPERIMENT #044 - MTF Donchian+RSI+ATR 1h+4h Fixed State Management
==================================================================================================
Hypothesis: Donchian breakouts with 4h trend filter capture sustained moves while RSI avoids
false breakouts at extremes. 1h+4h timeframe pair is more stable than 15m+4h (less noise).
Previous crashes (#040-#043) were from numpy array mutation - using pure Python lists for state.

Key changes from #043:
- Timeframe: 1h entries + 4h trend (proven stable combination from best strategy)
- Indicators: Donchian Channel (breakout), RSI (momentum filter), ATR (volatility/stop)
- Position sizing: Base 0.25 with ATR dynamic (0.18-0.30 range)
- Stoploss: 2.5*ATR (wider to avoid noise exits)
- Take profit: 3R with trail at 1.5R (let winners run)
- State tracking: 100% Python lists (no numpy mutation crashes)

Why this should work:
- Donchian breakouts capture trend continuation (20-period high/low)
- 4h trend filter ensures we trade with higher timeframe momentum
- RSI filter avoids entering at overbought/oversold extremes (false breakouts)
- Wider stops (2.5*ATR) reduce premature exits in volatile crypto markets
- 1h timeframe balances responsiveness with noise reduction vs 15m
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf


name = "mtf_donchian_rsi_atr_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, middle


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # Initial SMA
    avg_gain[period - 1] = np.mean(gain[:period])
    avg_loss[period - 1] = np.mean(loss[:period])
    
    # Wilder's smoothing
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)
    
    return rsi


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
    atr[period - 1] = np.mean(tr[:period])
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return np.nan_to_num(atr, nan=0.0)


def calculate_sma(close, period=20):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return np.nan_to_num(sma, nan=0.0)


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
    rsi_4h = calculate_rsi(close_4h, period=14)
    sma_4h = calculate_sma(close_4h, period=50)
    
    # Align 4h indicators to 1h timeframe (auto shift(1) for completed bars)
    donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    sma_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_4h)
    
    # 1h entry indicators
    donchian_upper_1h, donchian_lower_1h, donchian_mid_1h = calculate_donchian(high, low, period=20)
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_1h = calculate_sma(close, period=50)
    
    # Parameters
    BASE_SIZE = 0.25
    MIN_SIZE = 0.18
    MAX_SIZE = 0.30
    ATR_STOP_MULT = 2.5  # Wider stops for crypto volatility
    TP_MULT = 3.0  # Let winners run
    TRAIL_MULT = 1.5  # Trail after TP hit
    RSI_LONG_MIN = 35  # Avoid oversold breakouts (weak momentum)
    RSI_LONG_MAX = 70  # Avoid overbought entries
    RSI_SHORT_MIN = 30  # Avoid oversold shorts
    RSI_SHORT_MAX = 65  # Avoid overbought breakouts (weak momentum)
    
    # Use Python lists for ALL mutable state (CRITICAL - avoids numpy read-only crash)
    signals_list = [0.0] * n
    position_state = [0.0] * n  # 0=flat, 1=long, -1=short
    entry_price_list = [0.0] * n
    tp_triggered_list = [0.0] * n
    extreme_price_list = [0.0] * n
    
    first_valid = max(150, 50 * 4)  # Ensure 4h data is aligned and indicators warmed up
    
    for i in range(first_valid, n):
        # Validate data
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 0:
            position_state[i] = 0.0
            entry_price_list[i] = 0.0
            tp_triggered_list[i] = 0.0
            extreme_price_list[i] = 0.0
            signals_list[i] = 0.0
            continue
        
        if np.isnan(donchian_mid_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]):
            position_state[i] = 0.0
            entry_price_list[i] = 0.0
            tp_triggered_list[i] = 0.0
            extreme_price_list[i] = 0.0
            signals_list[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        
        # 4h trend direction (price vs SMA50 + Donchian midpoint)
        trend_4h = 0
        if price > sma_4h_aligned[i] and price > donchian_mid_4h_aligned[i]:
            trend_4h = 1
        elif price < sma_4h_aligned[i] and price < donchian_mid_4h_aligned[i]:
            trend_4h = -1
        
        # 4h RSI momentum filter
        rsi_4h_val = rsi_4h_aligned[i]
        momentum_ok_long = RSI_LONG_MIN <= rsi_4h_val <= RSI_LONG_MAX
        momentum_ok_short = RSI_SHORT_MIN <= rsi_4h_val <= RSI_SHORT_MAX
        
        # Manage existing positions
        if position_state[i - 1] != 0:
            prev_side = position_state[i - 1]
            prev_entry = entry_price_list[i - 1] if entry_price_list[i - 1] > 0 else price
            prev_tp = tp_triggered_list[i - 1]
            prev_extreme = extreme_price_list[i - 1] if extreme_price_list[i - 1] > 0 else prev_entry
            
            # Update extreme price
            if prev_side > 0:
                current_extreme = max(prev_extreme, price)
            else:
                current_extreme = min(prev_extreme, price) if prev_extreme > 0 else price
            extreme_price_list[i] = current_extreme
            
            # Stoploss check
            exited = False
            if prev_side > 0:
                stop_price = prev_entry - ATR_STOP_MULT * atr
                if price < stop_price:
                    position_state[i] = 0.0
                    entry_price_list[i] = 0.0
                    tp_triggered_list[i] = 0.0
                    extreme_price_list[i] = 0.0
                    signals_list[i] = 0.0
                    exited = True
                
                # Take profit at 3R
                if not exited and not prev_tp:
                    tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                    if price >= tp_price:
                        position_state[i] = 1.0
                        entry_price_list[i] = prev_entry
                        tp_triggered_list[i] = 1.0
                        extreme_price_list[i] = current_extreme
                        signals_list[i] = BASE_SIZE * 0.5
                        exited = True
                
                # Trail stop at 1.5R after TP
                if not exited and prev_tp:
                    trail_price = current_extreme - TRAIL_MULT * ATR_STOP_MULT * atr
                    if price < trail_price:
                        position_state[i] = 0.0
                        entry_price_list[i] = 0.0
                        tp_triggered_list[i] = 0.0
                        extreme_price_list[i] = 0.0
                        signals_list[i] = 0.0
                        exited = True
            else:  # Short
                stop_price = prev_entry + ATR_STOP_MULT * atr
                if price > stop_price:
                    position_state[i] = 0.0
                    entry_price_list[i] = 0.0
                    tp_triggered_list[i] = 0.0
                    extreme_price_list[i] = 0.0
                    signals_list[i] = 0.0
                    exited = True
                
                if not exited and not prev_tp:
                    tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                    if price <= tp_price:
                        position_state[i] = -1.0
                        entry_price_list[i] = prev_entry
                        tp_triggered_list[i] = 1.0
                        extreme_price_list[i] = current_extreme
                        signals_list[i] = -BASE_SIZE * 0.5
                        exited = True
                
                if not exited and prev_tp:
                    trail_price = current_extreme + TRAIL_MULT * ATR_STOP_MULT * atr
                    if price > trail_price:
                        position_state[i] = 0.0
                        entry_price_list[i] = 0.0
                        tp_triggered_list[i] = 0.0
                        extreme_price_list[i] = 0.0
                        signals_list[i] = 0.0
                        exited = True
            
            # Hold position - copy previous state
            if not exited:
                signals_list[i] = signals_list[i - 1]
                position_state[i] = position_state[i - 1]
                entry_price_list[i] = entry_price_list[i - 1]
                tp_triggered_list[i] = tp_triggered_list[i - 1]
                extreme_price_list[i] = extreme_price_list[i - 1]
            continue
        
        # Entry logic: Donchian breakout + 4h trend + RSI momentum
        rsi_1h_val = rsi_1h[i]
        
        # Long entry: 4h uptrend + Donchian breakout + RSI momentum filter
        if trend_4h == 1 and price > donchian_upper_1h[i] and momentum_ok_long:
            if RSI_LONG_MIN <= rsi_1h_val <= RSI_LONG_MAX:
                # Dynamic sizing based on ATR
                atr_pct = atr / price
                vol_ratio = 0.012 / max(atr_pct, 0.001)
                vol_ratio = np.clip(vol_ratio, 0.7, 1.3)
                position_size = np.clip(BASE_SIZE * vol_ratio, MIN_SIZE, MAX_SIZE)
                
                position_state[i] = 1.0
                entry_price_list[i] = price
                tp_triggered_list[i] = 0.0
                extreme_price_list[i] = price
                signals_list[i] = position_size
                continue
        
        # Short entry: 4h downtrend + Donchian breakdown + RSI momentum filter
        elif trend_4h == -1 and price < donchian_lower_1h[i] and momentum_ok_short:
            if RSI_SHORT_MIN <= rsi_1h_val <= RSI_SHORT_MAX:
                # Dynamic sizing based on ATR
                atr_pct = atr / price
                vol_ratio = 0.012 / max(atr_pct, 0.001)
                vol_ratio = np.clip(vol_ratio, 0.7, 1.3)
                position_size = np.clip(BASE_SIZE * vol_ratio, MIN_SIZE, MAX_SIZE)
                
                position_state[i] = -1.0
                entry_price_list[i] = price
                tp_triggered_list[i] = 0.0
                extreme_price_list[i] = price
                signals_list[i] = -position_size
                continue
        
        # No position
        position_state[i] = 0.0
        entry_price_list[i] = 0.0
        tp_triggered_list[i] = 0.0
        extreme_price_list[i] = 0.0
        signals_list[i] = 0.0
    
    return np.array(signals_list)