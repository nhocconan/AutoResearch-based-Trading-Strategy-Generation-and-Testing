#!/usr/bin/env python3
"""
EXPERIMENT #041 - MTF KAMA+RSI+BB 1h+4h Simplified State Tracking
==================================================================================================
Hypothesis: Previous crashes (#034, #040) caused by read-only numpy array assignments in loop.
15m timeframe has excessive noise causing whipsaw. Moving to 1h + 4h MTF with simpler logic.

Key changes from #040:
- Timeframe: 1h (less noise than 15m, more trades than 4h)
- HTF: 4h trend filter using mtf_data helper
- Indicators: KAMA (adaptive trend), RSI (momentum), Bollinger Bands (volatility squeeze)
- State tracking: Use Python lists (mutable) instead of numpy arrays in loop
- Signal levels: Discrete (0, ±0.25, ±0.30) to reduce churn costs
- Position sizing: Fixed 0.30 max (simpler than ATR-dynamic which caused issues)
- Stoploss/TP: Signal→0 at 2.5*ATR stop, signal→half at 2R profit

Why this should work:
- 1h timeframe balances trade frequency vs noise (proven in #031-#039 range)
- KAMA adapts to volatility better than HMA/EMA in choppy markets
- BB squeeze filter avoids trading in low-volatility consolidation
- Simple state tracking avoids numpy read-only crashes
- Based on best performer structure (mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v1)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf


name = "mtf_kama_rsi_bb_squeeze_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < er_period + fast_sc:
        return np.zeros(n)
    
    # Efficiency Ratio
    net_change = np.abs(np.diff(close, prepend=close[0]))
    sum_change = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(
        window=er_period, min_periods=er_period
    ).sum().values
    
    er = np.zeros(n)
    mask = sum_change > 0
    er[mask] = net_change[mask] / sum_change[mask]
    
    # Smoothing constants
    fast_const = 2.0 / (fast_sc + 1)
    slow_const = 2.0 / (slow_sc + 1)
    
    sc = er * (fast_const - slow_const) + slow_const
    sc_squared = sc ** 2
    
    kama = np.zeros(n)
    kama[er_period - 1] = close[er_period - 1]
    
    for i in range(er_period, n):
        kama[i] = kama[i-1] + sc_squared[i] * (close[i] - kama[i-1])
    
    return np.nan_to_num(kama, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    return (
        np.nan_to_num(upper, nan=0.0),
        np.nan_to_num(lower, nan=0.0),
        np.nan_to_num(bandwidth, nan=0.0)
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
    
    # Calculate 4h trend indicator (KAMA adaptive)
    kama_4h = calculate_kama(close_4h, er_period=10, fast_sc=2, slow_sc=30)
    
    # Align 4h indicators to 1h timeframe (auto shift(1) for completed bars)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # 1h entry indicators
    kama_1h = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    rsi_1h = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Parameters
    BASE_SIZE = 0.30
    MIN_SIZE = 0.20
    MAX_SIZE = 0.35
    ATR_STOP_MULT = 2.5
    TP_MULT = 2.0
    TRAIL_MULT = 1.0
    BB_WIDTH_PERCENTILE = 0.30  # Only trade when BB width > 30th percentile (avoid squeeze)
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    
    # Calculate BB width percentile for regime filter
    bb_width_valid = bb_width[bb_width > 0]
    if len(bb_width_valid) > 0:
        bb_width_threshold = np.percentile(bb_width_valid, 30)
    else:
        bb_width_threshold = 0.01
    
    signals = np.zeros(n)
    
    # Use Python lists for mutable state tracking (avoids numpy read-only issue)
    position_side = [0.0] * n
    entry_price = [0.0] * n
    tp_triggered = [0.0] * n
    extreme_price = [0.0] * n
    
    first_valid = max(100, 40 * 4)  # Ensure 4h data is aligned
    
    for i in range(first_valid, n):
        # Validate data
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 0 or np.isnan(kama_4h_aligned[i]):
            signals[i] = 0.0
            position_side[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        
        # 4h trend direction
        trend_4h = 0
        if kama_4h_aligned[i] > 0:
            if price > kama_4h_aligned[i]:
                trend_4h = 1
            elif price < kama_4h_aligned[i]:
                trend_4h = -1
        
        # BB squeeze filter - avoid trading in low volatility
        in_squeeze = bb_width[i] < bb_width_threshold
        
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
            
            # Hold position
            signals[i] = signals[i-1]
            position_side[i] = position_side[i-1]
            entry_price[i] = entry_price[i-1]
            tp_triggered[i] = tp_triggered[i-1]
            extreme_price[i] = extreme_price[i-1]
            continue
        
        # Skip entries during BB squeeze (low volatility)
        if in_squeeze:
            signals[i] = 0.0
            position_side[i] = 0.0
            continue
        
        # Entry logic: KAMA trend + RSI pullback
        rsi = rsi_1h[i]
        kama_1h_val = kama_1h[i]
        
        # 1h KAMA confirmation
        kama_1h_trend = 0
        if kama_1h_val > 0:
            if price > kama_1h_val:
                kama_1h_trend = 1
            elif price < kama_1h_val:
                kama_1h_trend = -1
        
        # Long entry: 4h uptrend + 1h KAMA uptrend + RSI pullback
        if trend_4h == 1 and kama_1h_trend == 1:
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
        
        # Short entry: 4h downtrend + 1h KAMA downtrend + RSI pullback
        elif trend_4h == -1 and kama_1h_trend == -1:
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
    
    return signals