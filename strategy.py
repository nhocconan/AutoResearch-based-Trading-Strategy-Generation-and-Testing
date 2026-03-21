#!/usr/bin/env python3
"""
EXPERIMENT #005 - MTF KAMA+Donchian+RSI+Zscore (15m+1h+4h v1)
==================================================================================================
Hypothesis: Replace HMA with KAMA (Kaufman Adaptive Moving Average) for trend filtering.
KAMA adapts to market volatility - moves fast in trends, slow in chop. Combine with:
- 4h KAMA trend direction (volatility-adaptive trend filter)
- 1h Donchian breakout (momentum confirmation - price breaking N-period high/low)
- 15m RSI + Z-score for entry timing (mean reversion within trend)

Why this should beat current best:
- KAMA responds better to regime changes than HMA (less whipsaw in chop)
- Donchian breakout confirms momentum is actually breaking out (vs just RSI pullback)
- Z-score adds statistical filter for extreme moves (avoids chasing)
- Same 3-TF structure as proven winner but different signal types

Position sizing: 0.30 max (conservative), discrete levels (0.0, ±0.20, ±0.30)
Stoploss: 2.0*ATR trailing, Take profit: 2R then trail at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_donchian_rsi_zscore_15m_1h_4h_v1"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - moves fast in trends, slow in chop
    """
    n = len(close)
    if n < period + slow_period:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        direction = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[period] = close[period]  # Initialize with price
    
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
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over N periods)"""
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
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    
    # Get 1h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        h_1h = df_1h['high'].values
        l_1h = df_1h['low'].values
        
        # 1h Donchian for breakout confirmation
        donchian_upper_1h, donchian_lower_1h = calculate_donchian(h_1h, l_1h, period=20)
        
        # Align 1h indicators to 15m timeframe (auto shift for completed bars)
        donchian_upper_1h_aligned = align_htf_to_ltf(prices, df_1h, donchian_upper_1h)
        donchian_lower_1h_aligned = align_htf_to_ltf(prices, df_1h, donchian_lower_1h)
    except Exception:
        # Fallback if mtf_data fails
        donchian_upper_1h_aligned = np.zeros(n)
        donchian_lower_1h_aligned = np.zeros(n)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        
        # 4h KAMA for adaptive trend
        kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
        
        # Align 4h indicators to 15m timeframe
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        
        # Calculate 4h trend direction (price vs KAMA)
        trend_4h = np.zeros(n)
        for i in range(n):
            if i < len(kama_4h_aligned) and kama_4h_aligned[i] > 0:
                if close[i] > kama_4h_aligned[i]:
                    trend_4h[i] = 1
                elif close[i] < kama_4h_aligned[i]:
                    trend_4h[i] = -1
    except Exception:
        kama_4h_aligned = np.zeros(n)
        trend_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score threshold (avoid extreme moves - chasing)
    ZSCORE_MAX = 1.5
    
    # Donchian breakout confirmation (price near channel edge)
    DONCHIAN_BREAKOUT_PCT = 0.02  # Within 2% of channel edge
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 2, 20, 30 + 10)
    
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
        
        # Get aligned MTF values
        trend_4h_val = trend_4h[i] if i < len(trend_4h) else 0
        donchian_upper_1h = donchian_upper_1h_aligned[i] if i < len(donchian_upper_1h_aligned) else 0
        donchian_lower_1h = donchian_lower_1h_aligned[i] if i < len(donchian_lower_1h_aligned) else 0
        zscore_val = zscore_15m[i] if i < len(zscore_15m) else 0
        
        # Z-score filter - avoid extreme moves (chasing)
        if abs(zscore_val) > ZSCORE_MAX:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                position_side[i] = 0
            else:
                position_side[i] = 0
            continue
        
        # 4h trend filter (price vs KAMA)
        if trend_4h_val == 0:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                position_side[i] = 0
            else:
                position_side[i] = 0
            continue
        
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
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_15m[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_15m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_15m[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_15m[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_15m[i]
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
        
        # Entry logic: 4h KAMA trend + 1h Donchian breakout + 15m RSI pullback + Z-score
        price = close[i]
        
        if trend_4h_val == 1:  # Bullish trend on 4h (price > KAMA)
            # Donchian breakout confirmation (price near upper channel on 1h)
            # RSI pullback on 15m (not overbought)
            # Z-score not extreme
            if (donchian_upper_1h > 0 and 
                price >= donchian_upper_1h * (1 - DONCHIAN_BREAKOUT_PCT) and
                RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX and
                zscore_val < ZSCORE_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend_4h_val == -1:  # Bearish trend on 4h (price < KAMA)
            # Donchian breakout confirmation (price near lower channel on 1h)
            # RSI pullback on 15m (not oversold)
            # Z-score not extreme
            if (donchian_lower_1h > 0 and 
                price <= donchian_lower_1h * (1 + DONCHIAN_BREAKOUT_PCT) and
                RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX and
                zscore_val > -ZSCORE_MAX):
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals