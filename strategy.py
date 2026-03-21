#!/usr/bin/env python3
"""
EXPERIMENT #034 - MTF KAMA+Donchian+RSI (30m+4h v1)
==================================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA/EMA.
Combined with Donchian channels for breakout confirmation and 4h trend filter.
This differs from current best (mtf_hma_rsi_30m_4h_simplified_v2) by:
- KAMA instead of HMA (adaptive to market regime)
- Donchian channel breakout confirmation (clearer than simple MA cross)
- Same 30m base + 4h trend (proven combination)
- Simpler position management to ensure ≥10 trades

Why this should work:
- KAMA flattens in chop, steepens in trend (reduces whipsaws)
- Donchian 20-period breakout confirms trend direction
- 4h trend filter prevents counter-trend trades
- 30m timeframe has proven success (current best uses 30m)
- Position size 0.30 (conservative, controls drawdown)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_donchian_rsi_30m_4h_v1"
timeframe = "30m"
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
    KAMA adapts to market volatility - flattens in chop, steepens in trend
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[max(0, i - period):i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 30m indicators for entry timing
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    kama_30m = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    donchian_upper_30m, donchian_lower_30m = calculate_donchian(high, low, period=20)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
        
        # 4h Donchian for trend confirmation
        donchian_upper_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
        
        # Align 4h indicators to 30m timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
        donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        donchian_upper_4h_aligned = np.zeros(n)
        donchian_lower_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 14 * 2, 20, 30 + 10)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_30m[i]) or np.isnan(rsi_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_30m[i]) or kama_30m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        kama_4h_val = kama_4h_aligned[i] if i < len(kama_4h_aligned) else 0
        donchian_upper_4h_val = donchian_upper_4h_aligned[i] if i < len(donchian_upper_4h_aligned) else 0
        donchian_lower_4h_val = donchian_lower_4h_aligned[i] if i < len(donchian_lower_4h_aligned) else 0
        
        # Skip if 4h data not available
        if kama_4h_val == 0 or donchian_upper_4h_val == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filter: price vs KAMA + Donchian position
        price_4h_idx = min(i // 8, len(c_4h) - 1)  # 8 x 30m = 4h
        if price_4h_idx >= 0 and price_4h_idx < len(c_4h):
            price_4h = c_4h[price_4h_idx]
            
            # Bullish trend: price > KAMA and in upper half of Donchian
            mid_4h = (donchian_upper_4h_val + donchian_lower_4h_val) / 2
            trend_4h_bullish = (price_4h > kama_4h_val) and (price_4h > mid_4h)
            trend_4h_bearish = (price_4h < kama_4h_val) and (price_4h < mid_4h)
        else:
            trend_4h_bullish = False
            trend_4h_bearish = False
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_30m[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_30m[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_30m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_30m[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_30m[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_30m[i]
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
        
        # Entry logic: 4h trend + 30m KAMA + RSI + Donchian breakout
        price = close[i]
        
        # Long entry: 4h bullish + 30m price > KAMA + RSI in range + breakout above Donchian mid
        donchian_mid_30m = (donchian_upper_30m[i] + donchian_lower_30m[i]) / 2
        
        if trend_4h_bullish:
            if (price > kama_30m[i] and
                RSI_LONG_MIN <= rsi_30m[i] <= RSI_LONG_MAX and
                price > donchian_mid_30m):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        # Short entry: 4h bearish + 30m price < KAMA + RSI in range + breakout below Donchian mid
        elif trend_4h_bearish:
            if (price < kama_30m[i] and
                RSI_SHORT_MIN <= rsi_30m[i] <= RSI_SHORT_MAX and
                price < donchian_mid_30m):
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals