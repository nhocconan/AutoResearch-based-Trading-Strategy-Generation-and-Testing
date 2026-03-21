#!/usr/bin/env python3
"""
EXPERIMENT #026 - KAMA+RSI+ATR Dynamic Sizing (30m+4h v1)
==================================================================================================
Hypothesis: Simplify the MTF approach using 30m base + 4h trend (like current best #022) but with:
- KAMA instead of HMA (adaptive to volatility, less whipsaw in chop)
- ATR-based dynamic position sizing (size adjusts to current volatility)
- Simpler entry logic (fewer filters = more trades, avoid over-filtering)
- Cleaner stoploss/takeprofit with trailing

Why this should beat #004 and #022:
- 30m/4h combination proven in #022 (Sharpe=1.153)
- KAMA adapts to market regime better than fixed HMA
- Dynamic sizing reduces risk in high volatility periods
- Fewer filters means more trades (avoid the 0-trade failures like #014, #024)
- Discrete signal levels (0.0, ±0.25, ±0.35) reduce churn costs
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_atr_dynamic_30m_4h_v1"
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
    Kaufman's Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = sum(abs(close[j] - close[j - 1]) for j in range(i - period + 1, i + 1))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
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
    
    # 30m indicators for entry timing
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    kama_30m = calculate_kama(close, period=10, fast=2, slow=30)
    _, st_direction_30m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # Get 4h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(c_4h, period=10, fast=2, slow=30)
        _, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        
        # Align 4h indicators to 30m timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        kama_4h_aligned = np.zeros(n)
        st_direction_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.30  # Base position size (30% of capital)
    SIZE_HALF = 0.15
    
    # ATR-based dynamic sizing parameters
    TARGET_ATR_PCT = 0.02  # Target 2% ATR as baseline
    MAX_SIZE = 0.35  # Never exceed 35% position
    MIN_SIZE = 0.20  # Never go below 20% when in trade
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    ATR_TRAIL_MULT = 1.5
    
    # Minimum bars for warmup
    first_valid = max(100, 30, 14 * 2)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_30m[i]) or np.isnan(rsi_30m[i]) or atr_30m[i] == 0 or np.isnan(kama_30m[i]):
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        st_trend_4h = st_direction_4h_aligned[i] if i < len(st_direction_4h_aligned) else 0
        kama_4h_val = kama_4h_aligned[i] if i < len(kama_4h_aligned) else 0
        
        # Calculate 4h trend direction (price vs KAMA)
        trend_4h = 0
        if kama_4h_val > 0 and i < len(c_4h):
            idx_4h = min(i // 8, len(c_4h) - 1)  # 8 x 30m = 4h
            if idx_4h < len(c_4h) and idx_4h >= 0:
                if c_4h[idx_4h] > kama_4h[idx_4h]:
                    trend_4h = 1
                elif c_4h[idx_4h] < kama_4h[idx_4h]:
                    trend_4h = -1
        
        # 4h trend filter (Supertrend + KAMA must agree)
        trend_confirmed = False
        if st_trend_4h == 1 and trend_4h == 1:
            trend_confirmed = True
            trend_dir = 1
        elif st_trend_4h == -1 and trend_4h == -1:
            trend_confirmed = True
            trend_dir = -1
        
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
            
            # Calculate dynamic stoploss distance
            stop_distance = ATR_STOP_MULT * atr_30m[i]
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - stop_distance
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * stop_distance
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit after TP hit
                if prev_tp:
                    trail_stop = current_high - ATR_TRAIL_MULT * atr_30m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + stop_distance
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * stop_distance
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit after TP hit
                if prev_tp:
                    trail_stop = current_low + ATR_TRAIL_MULT * atr_30m[i]
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
        
        # No existing position - check for new entries
        price = close[i]
        
        # Calculate ATR-based dynamic position size
        atr_pct = atr_30m[i] / price if price > 0 else 0.02
        size_multiplier = TARGET_ATR_PCT / atr_pct if atr_pct > 0 else 1.0
        size_multiplier = np.clip(size_multiplier, MIN_SIZE / BASE_SIZE, MAX_SIZE / BASE_SIZE)
        dynamic_size = BASE_SIZE * size_multiplier
        dynamic_size = np.clip(dynamic_size, MIN_SIZE, MAX_SIZE)
        
        # Entry logic: 4h trend + 30m KAMA + 30m RSI pullback
        if trend_confirmed and trend_dir == 1:  # Bullish trend on 4h
            # Price above 30m KAMA (short-term bullish)
            # RSI pullback on 30m (not overbought)
            # Supertrend confirms direction
            if (price > kama_30m[i] and 
                st_direction_30m[i] == 1 and
                RSI_LONG_MIN <= rsi_30m[i] <= RSI_LONG_MAX):
                signals[i] = dynamic_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_confirmed and trend_dir == -1:  # Bearish trend on 4h
            # Price below 30m KAMA (short-term bearish)
            # RSI pullback on 30m (not oversold)
            # Supertrend confirms direction
            if (price < kama_30m[i] and 
                st_direction_30m[i] == -1 and
                RSI_SHORT_MIN <= rsi_30m[i] <= RSI_SHORT_MAX):
                signals[i] = -dynamic_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals