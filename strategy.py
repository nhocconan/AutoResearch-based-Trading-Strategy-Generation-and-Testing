#!/usr/bin/env python3
"""
EXPERIMENT #023 - KAMA Adaptive Trend + RSI Pullback (1h+4h)
==================================================================================================
Hypothesis: Replace HMA with KAMA (Kaufman Adaptive Moving Average) for better noise filtering.
KAMA adapts to market volatility - moves fast in trends, slow in chop. This should reduce
whipsaws compared to HMA while maintaining trend-following capability.

Key changes from current best (#022):
- Base timeframe: 1h (cleaner signals than 30m, more trades than 4h)
- Trend indicator: KAMA instead of HMA (adaptive to volatility)
- Position size: 0.30 max (more conservative than 0.35)
- Stoploss: 2.0*ATR (slightly tighter for better risk control)
- Remove complex state tracking - use simpler signal-based exits

Why this should beat Sharpe=1.153:
- KAMA reduces false signals in ranging markets (common in crypto)
- 1h has proven success in other experiments with cleaner signals
- Tighter stops should reduce max drawdown
- Simpler logic = fewer bugs
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_1h_4h_adaptive_v1"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - fast in trends, slow in chop
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[period] = close[period]  # Initialize
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
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
    
    for i in range(n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period - 1] = lower_band[period - 1]
    
    for i in range(period, n):
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
        
        # 4h Supertrend for trend confirmation
        _, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        
        # Align 4h indicators to 1h timeframe
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        st_direction_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45
    RSI_SHORT_ENTRY = 55
    RSI_EXIT_LONG = 65
    RSI_EXIT_SHORT = 35
    
    # ATR stoploss multiplier (tighter than previous)
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 40)  # KAMA needs more warmup
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1h[i]) or kama_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned 4h values
        kama_4h_val = kama_4h_aligned[i] if i < len(kama_4h_aligned) else 0
        st_4h_val = st_direction_4h_aligned[i] if i < len(st_direction_4h_aligned) else 0
        
        # Determine 4h trend direction using KAMA slope
        trend_4h = 0
        if i >= first_valid + 4 and kama_4h_val > 0:
            idx_4h = min(i // 4, len(kama_4h_aligned) - 1)
            if idx_4h >= 4 and idx_4h < len(kama_4h):
                if kama_4h[idx_4h] > kama_4h[idx_4h - 4]:
                    trend_4h = 1
                elif kama_4h[idx_4h] < kama_4h[idx_4h - 4]:
                    trend_4h = -1
        
        # Check existing positions first (stoploss and take profit)
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            
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
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if price >= tp_price and signals[i - 1] == SIZE_FULL:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R after TP
                if signals[i - 1] == SIZE_HALF:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # RSI exit signal
                if rsi_1h[i] >= RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if price <= tp_price and signals[i - 1] == -SIZE_FULL:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R after TP
                if signals[i - 1] == -SIZE_HALF:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # RSI exit signal
                if rsi_1h[i] <= RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 1h RSI pullback + 1h KAMA confirmation
        price = close[i]
        
        # Long entry: 4h bullish + 1h RSI pullback + price above 1h KAMA
        if trend_4h == 1 and st_4h_val == 1:
            if rsi_1h[i] <= RSI_LONG_ENTRY and rsi_1h[i] >= 30:
                if price > kama_1h[i]:  # Price above KAMA confirms trend
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    
        # Short entry: 4h bearish + 1h RSI pullback + price below 1h KAMA
        elif trend_4h == -1 and st_4h_val == -1:
            if rsi_1h[i] >= RSI_SHORT_ENTRY and rsi_1h[i] <= 70:
                if price < kama_1h[i]:  # Price below KAMA confirms trend
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals