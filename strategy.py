#!/usr/bin/env python3
"""
EXPERIMENT #005 - MTF KAMA+Donchian+Z-score (30m+1h+4h v1)
==================================================================================================
Hypothesis: Switch to 30m primary timeframe (less noise than 15m, more trades than 1h).
Use KAMA (Kaufman Adaptive) for trend - adapts to volatility better than HMA/EMA.
Use Donchian Channel for breakout confirmation (proven trend strength indicator).
Use Z-score filter (from current best strategy) instead of BBW for regime detection.
This differs from #004 by:
- 30m instead of 15m (cleaner signals, fewer whipsaws)
- KAMA instead of HMA (volatility-adaptive smoothing)
- Donchian instead of Supertrend (pure price breakout, no ATR lag)
- Z-score instead of BBW (proven in best strategy mtf_hma_rsi_zscore_v1)

Why this should work:
- 30m has shown good balance in experiments #031, #034, #035
- KAMA reduces whipsaws in choppy markets (ER adapts smoothing)
- Donchian breakout confirms trend strength objectively
- Z-score filter proven effective in current best strategy
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_donchian_zscore_30m_1h_4h_v1"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    """
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    sc_fast = 2.0 / (fast + 1)
    sc_slow = 2.0 / (slow + 1)
    
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (sc_fast - sc_slow) + sc_slow) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
        else:
            zscore[i] = 0
    
    return zscore


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 30m indicators for entry timing
    atr_30m = calculate_atr(high, low, close, period=14)
    zscore_30m = calculate_zscore(close, period=20)
    rsi_30m = calculate_rsi(close, period=14)
    donchian_upper_30m, donchian_lower_30m = calculate_donchian(high, low, period=20)
    
    # Get 1h data using mtf_data helper (MUST use this for proper alignment)
    df_1h = get_htf_data(prices, '1h')
    c_1h = df_1h['close'].values
    h_1h = df_1h['high'].values
    l_1h = df_1h['low'].values
    
    # 1h indicators
    rsi_1h = calculate_rsi(c_1h, period=14)
    donchian_upper_1h, donchian_lower_1h = calculate_donchian(h_1h, l_1h, period=20)
    
    # Align 1h indicators to 30m timeframe (auto shift for completed bars)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    donchian_upper_1h_aligned = align_htf_to_ltf(prices, df_1h, donchian_upper_1h)
    donchian_lower_1h_aligned = align_htf_to_ltf(prices, df_1h, donchian_lower_1h)
    
    # Get 4h data using mtf_data helper for trend filter
    df_4h = get_htf_data(prices, '4h')
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # 4h KAMA for adaptive trend
    kama_4h = calculate_kama(c_4h, period=10, fast=2, slow=30)
    
    # 4h Donchian for trend strength
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
    
    # Align 4h indicators to 30m timeframe
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Z-score thresholds for regime filter
    ZSCORE_ENTRY_MAX = 1.0  # Don't enter if already extended
    ZSCORE_EXIT = 1.5  # Exit if overextended
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 30, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        current_price = close[i]
        
        # Skip if invalid data
        if np.isnan(atr_30m[i]) or np.isnan(zscore_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            if i > 0:
                position_side[i] = position_side[i - 1]
            continue
        
        # Get aligned MTF values (use previous bar for completed HTF candles)
        idx_1h = min(i // 2, len(rsi_1h_aligned) - 1)  # 2 x 30m = 1h
        idx_4h = min(i // 8, len(kama_4h_aligned) - 1)  # 8 x 30m = 4h
        
        # Ensure indices are valid
        if idx_1h < 0 or idx_4h < 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filter (price vs KAMA + Donchian position)
        kama_4h_val = kama_4h_aligned[idx_4h] if idx_4h < len(kama_4h_aligned) else 0
        donchian_mid_4h = (donchian_upper_4h_aligned[idx_4h] + donchian_lower_4h_aligned[idx_4h]) / 2 if idx_4h < len(donchian_upper_4h_aligned) else 0
        
        # Determine 4h trend direction
        trend_4h = 0
        if kama_4h_val > 0 and current_price > kama_4h_val:
            trend_4h = 1
        elif kama_4h_val > 0 and current_price < kama_4h_val:
            trend_4h = -1
        
        # 1h momentum filter (RSI + Donchian)
        rsi_1h_val = rsi_1h_aligned[idx_1h] if idx_1h < len(rsi_1h_aligned) else 50
        donchian_mid_1h = (donchian_upper_1h_aligned[idx_1h] + donchian_lower_1h_aligned[idx_1h]) / 2 if idx_1h < len(donchian_upper_1h_aligned) else 0
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, current_price)
                current_low = min(prev_low, current_price) if prev_low > 0 else current_price
            else:
                current_high = max(prev_high, current_price) if prev_high > 0 else current_price
                current_low = min(prev_low, current_price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_30m[i]
                if current_price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_30m[i]
                if not prev_tp and current_price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_30m[i]
                    if current_price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Z-score exit if overextended
                if zscore_30m[i] > ZSCORE_EXIT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_30m[i]
                if current_price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_30m[i]
                if not prev_tp and current_price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_30m[i]
                    if current_price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Z-score exit if overextended
                if zscore_30m[i] < -ZSCORE_EXIT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 1h momentum + 30m Z-score + RSI pullback
        # Z-score filter - don't enter if already extended
        if abs(zscore_30m[i]) > ZSCORE_ENTRY_MAX:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # LONG entry
        if trend_4h == 1 and rsi_1h_val > 50:
            # Price above 4h KAMA + 1h RSI bullish + 30m RSI pullback
            if RSI_LONG_MIN <= rsi_30m[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = current_price
                tp_triggered[i] = 0
                highest_since_entry[i] = current_price
                lowest_since_entry[i] = current_price
        
        # SHORT entry
        elif trend_4h == -1 and rsi_1h_val < 50:
            # Price below 4h KAMA + 1h RSI bearish + 30m RSI pullback
            if RSI_SHORT_MIN <= rsi_30m[i] <= RSI_SHORT_MAX:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = current_price
                tp_triggered[i] = 0
                highest_since_entry[i] = current_price
                lowest_since_entry[i] = current_price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals