#!/usr/bin/env python3
"""
EXPERIMENT #043 - MTF KAMA+StochRSI+BBW 15m+4h Fixed State Management
==================================================================================================
Hypothesis: Previous crashes (#040-#042) caused by numpy array mutation in loop.
This version uses PURE Python lists for ALL state tracking - never modify numpy arrays.
KAMA adapts to volatility, StochRSI gives precise entry timing, BBW filters volatility regime.

Key changes from #042:
- State tracking: 100% Python lists (position_state, entry_price, tp_hit, etc.)
- Signal assignment: Only write signals[i] once per iteration, never modify in-place
- Timeframe: 15m entries + 4h trend (more responsive than 1h+4h)
- Indicators: KAMA (adaptive), StochRSI (momentum), BBW (volatility regime)
- Position sizing: Base 0.25 with ATR dynamic (0.18-0.30 range)
- Stoploss: 2.0*ATR with TP at 2R, trail at 1R

Why this should work:
- KAMA adapts to market conditions better than static EMAs
- StochRSI provides cleaner overbought/oversold signals than RSI
- BBW filter avoids trading during extreme volatility (major DD source)
- Pure Python state tracking avoids numpy read-only crashes
- Conservative sizing protects against 2022-style crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf


name = "mtf_kama_stochrsi_bbw_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    volatility = np.zeros(n)
    for i in range(er_period, n):
        vol_sum = 0.0
        for j in range(1, er_period + 1):
            vol_sum += np.abs(close[i - j + 1] - close[i - j])
        volatility[i] = vol_sum if vol_sum > 0 else 0.0001
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.nan_to_num(er, nan=0.0)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    sc[er_period:] = (er[er_period:] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama


def calculate_stoch_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """Calculate Stochastic RSI"""
    n = len(close)
    if n < rsi_period + stoch_period:
        return np.zeros(n), np.zeros(n)
    
    # Calculate RSI first
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    # Wilder's smoothing for RSI
    for i in range(rsi_period, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)
    
    # Calculate Stochastic of RSI
    stoch_rsi = np.zeros(n)
    for i in range(stoch_period, n):
        rsi_low = np.min(rsi[i - stoch_period + 1:i + 1])
        rsi_high = np.max(rsi[i - stoch_period + 1:i + 1])
        if rsi_high > rsi_low:
            stoch_rsi[i] = 100 * (rsi[i] - rsi_low) / (rsi_high - rsi_low)
        else:
            stoch_rsi[i] = 50.0
    
    # %K and %D
    k_line = pd.Series(stoch_rsi).rolling(window=k_period, min_periods=k_period).mean().values
    d_line = pd.Series(k_line).rolling(window=d_period, min_periods=d_period).mean().values
    
    return np.nan_to_num(k_line, nan=50.0), np.nan_to_num(d_line, nan=50.0)


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bb_width = (upper - lower) / middle * 100  # Band width as percentage
    
    return (
        np.nan_to_num(upper, nan=0.0),
        np.nan_to_num(lower, nan=0.0),
        np.nan_to_num(middle, nan=0.0),
        np.nan_to_num(bb_width, nan=0.0)
    )


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
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
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
    kama_4h_fast = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
    kama_4h_slow = calculate_kama(close_4h, er_period=10, fast_period=5, slow_period=50)
    bb_upper_4h, bb_lower_4h, bb_mid_4h, bbw_4h = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
    
    # Align 4h indicators to 15m timeframe (auto shift(1) for completed bars)
    kama_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_fast)
    kama_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_slow)
    bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
    
    # 15m entry indicators
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    stoch_k_15m, stoch_d_15m = calculate_stoch_rsi(close, rsi_period=14, stoch_period=14)
    bb_upper_15m, bb_lower_15m, bb_mid_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    atr_15m = calculate_atr(high, low, close, period=14)
    
    # Parameters
    BASE_SIZE = 0.25
    MIN_SIZE = 0.18
    MAX_SIZE = 0.30
    ATR_STOP_MULT = 2.0
    TP_MULT = 2.0
    TRAIL_MULT = 1.0
    BBW_MIN = 2.0  # Avoid extremely low volatility
    BBW_MAX = 15.0  # Avoid extremely high volatility
    STCH_LONG_THRESHOLD = 20  # StochRSI oversold for long entry
    STCH_SHORT_THRESHOLD = 80  # StochRSI overbought for short entry
    
    # Use Python lists for ALL mutable state (CRITICAL - avoids numpy read-only crash)
    signals_list = [0.0] * n
    position_state = [0.0] * n  # 0=flat, 1=long, -1=short
    entry_price_list = [0.0] * n
    tp_triggered_list = [0.0] * n
    extreme_price_list = [0.0] * n
    
    first_valid = max(150, 50 * 4)  # Ensure 4h data is aligned and indicators warmed up
    
    for i in range(first_valid, n):
        # Validate data
        if np.isnan(atr_15m[i]) or atr_15m[i] <= 0:
            position_state[i] = 0.0
            entry_price_list[i] = 0.0
            tp_triggered_list[i] = 0.0
            extreme_price_list[i] = 0.0
            signals_list[i] = 0.0
            continue
        
        if np.isnan(kama_4h_fast_aligned[i]) or np.isnan(bbw_4h_aligned[i]):
            position_state[i] = 0.0
            entry_price_list[i] = 0.0
            tp_triggered_list[i] = 0.0
            extreme_price_list[i] = 0.0
            signals_list[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        
        # 4h trend direction (KAMA fast vs slow)
        trend_4h = 0
        if kama_4h_fast_aligned[i] > kama_4h_slow_aligned[i]:
            trend_4h = 1
        elif kama_4h_fast_aligned[i] < kama_4h_slow_aligned[i]:
            trend_4h = -1
        
        # 4h volatility regime filter (BB Width)
        bbw = bbw_4h_aligned[i]
        volatility_ok = BBW_MIN <= bbw <= BBW_MAX
        
        # Manage existing positions
        if position_state[i-1] != 0:
            prev_side = position_state[i-1]
            prev_entry = entry_price_list[i-1] if entry_price_list[i-1] > 0 else price
            prev_tp = tp_triggered_list[i-1]
            prev_extreme = extreme_price_list[i-1] if extreme_price_list[i-1] > 0 else prev_entry
            
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
                
                # Take profit at 2R
                if not exited and not prev_tp:
                    tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                    if price >= tp_price:
                        position_state[i] = 1.0
                        entry_price_list[i] = prev_entry
                        tp_triggered_list[i] = 1.0
                        extreme_price_list[i] = current_extreme
                        signals_list[i] = BASE_SIZE * 0.5
                        exited = True
                
                # Trail stop at 1R after TP
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
                signals_list[i] = signals_list[i-1]
                position_state[i] = position_state[i-1]
                entry_price_list[i] = entry_price_list[i-1]
                tp_triggered_list[i] = tp_triggered_list[i-1]
                extreme_price_list[i] = extreme_price_list[i-1]
            continue
        
        # Skip entries when volatility is extreme
        if not volatility_ok:
            position_state[i] = 0.0
            entry_price_list[i] = 0.0
            tp_triggered_list[i] = 0.0
            extreme_price_list[i] = 0.0
            signals_list[i] = 0.0
            continue
        
        # Entry logic: KAMA trend + StochRSI momentum + BBW regime
        stoch_k = stoch_k_15m[i]
        stoch_d = stoch_d_15m[i]
        
        # Long entry: 4h uptrend + StochRSI oversold + price above KAMA
        if trend_4h == 1 and stoch_k < STCH_LONG_THRESHOLD and stoch_k > stoch_d:
            if price > kama_15m[i]:
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
        
        # Short entry: 4h downtrend + StochRSI overbought + price below KAMA
        elif trend_4h == -1 and stoch_k > STCH_SHORT_THRESHOLD and stoch_k < stoch_d:
            if price < kama_15m[i]:
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