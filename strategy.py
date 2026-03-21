#!/usr/bin/env python3
"""
EXPERIMENT #043 - KAMA MACD Momentum with Multi-TF Trend Filter (1h Primary)
==================================================================================================
Hypothesis: Current best (Sharpe=0.537) uses 4h HMA+RSI pullback. This tries 1h primary with
KAMA adaptive trend + MACD momentum entries + 4h/1d trend filters.

Key innovations:
1. 1h PRIMARY + 4h/1d HTF: More trades than 4h, cleaner than 15m/30m
2. KAMA for trend: Adaptive MA that speeds up in trends, slows in ranges (Kaufman's ER)
3. MACD histogram momentum: Enter on histogram expansion in trend direction
4. ATR dynamic sizing: size = base * (target_vol / current_vol) for consistent risk
5. 1.5*ATR stoploss: Tighter than 2.0*ATR, with trailing at 1R

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- 1h timeframe captures more momentum moves than 4h
- KAMA adapts to volatility better than HMA
- MACD histogram expansion catches momentum earlier than RSI pullback
- Dynamic sizing reduces position in high volatility periods
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_macd_momentum_mtf_1h_4h_1d_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average
    Adapts to market noise using Efficiency Ratio
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    macd_line = (ema_fast - ema_slow).values
    signal_line = pd.Series(macd_line).ewm(span=signal_period, adjust=False, min_periods=signal_period).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    if n < len(atr) or len(atr) == 0:
        return np.zeros(n), np.zeros(n)
    
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        if atr[i] == 0:
            continue
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
    
    first_valid = np.where(atr > 0)[0]
    if len(first_valid) == 0:
        return supertrend, trend
    
    start_idx = first_valid[0]
    supertrend[start_idx] = upper_band[start_idx]
    trend[start_idx] = 1
    
    for i in range(start_idx + 1, n):
        if atr[i] == 0:
            supertrend[i] = supertrend[i - 1]
            trend[i] = trend[i - 1]
            continue
        
        if trend[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = max(supertrend[i - 1], lower_band[i])
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = min(supertrend[i - 1], upper_band[i])
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    macd_1h, signal_1h, hist_1h = calculate_macd(close, fast=12, slow=26, signal_period=9)
    supertrend_1h, st_trend_1h = calculate_supertrend(high, low, close, atr_1h, multiplier=3.0)
    
    # ========== 4h INDICATORS (INTERMEDIATE TREND) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        
        # Align to 1h timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        st_trend_4h_aligned = np.zeros(n)
    
    # ========== 1d INDICATORS (LONG-TERM TREND) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        kama_1d = calculate_kama(close_1d, er_period=10, fast_period=2, slow_period=30)
        atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
        _, st_trend_1d = calculate_supertrend(high_1d, low_1d, close_1d, atr_1d, multiplier=3.0)
        
        # Align to 1h timeframe (auto shift for completed bars)
        kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
        st_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, st_trend_1d)
        
    except Exception:
        kama_1d_aligned = np.zeros(n)
        st_trend_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DYNAMIC based on ATR
    SIZE_BASE = 0.25      # Base position (25% of capital)
    SIZE_HIGH = 0.35      # High conviction (35% of capital)
    TARGET_ATR_PCT = 0.02 # Target 2% ATR as % of price
    
    # Stoploss and take profit
    ATR_STOP_MULT = 1.5   # Tighter stoploss
    TP_MULT = 2.0         # 2R take profit
    TRAIL_MULT = 1.0      # Trail at 1R
    
    # MACD momentum thresholds
    HIST_THRESHOLD = 0.0  # Histogram must be expanding
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(kama_1h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        kama_val = kama_1h[i]
        hist_val = hist_1h[i]
        hist_prev = hist_1h[i - 1] if i > 0 else 0
        st_trend_val = st_trend_1h[i]
        
        # 4h trend filters
        kama_4h_val = kama_4h_aligned[i]
        st_trend_4h_val = st_trend_4h_aligned[i]
        
        # 1d trend filters
        kama_1d_val = kama_1d_aligned[i]
        st_trend_1d_val = st_trend_1d_aligned[i]
        
        # Determine trend directions
        trend_4h = 0
        if kama_4h_val > 0 and price > kama_4h_val:
            trend_4h = 1
        elif kama_4h_val > 0 and price < kama_4h_val:
            trend_4h = -1
        
        if st_trend_4h_val == 1:
            trend_4h = max(trend_4h, 1)
        elif st_trend_4h_val == -1:
            trend_4h = min(trend_4h, -1)
        
        trend_1d = 0
        if kama_1d_val > 0 and price > kama_1d_val:
            trend_1d = 1
        elif kama_1d_val > 0 and price < kama_1d_val:
            trend_1d = -1
        
        if st_trend_1d_val == 1:
            trend_1d = max(trend_1d, 1)
        elif st_trend_1d_val == -1:
            trend_1d = min(trend_1d, -1)
        
        # ========== CHECK EXISTING POSITIONS ==========
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
            
            # Stoploss check (1.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE / 2  # Reduce to half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - TRAIL_MULT * ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE / 2  # Reduce to half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + TRAIL_MULT * ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
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
        
        # ========== ENTRY LOGIC - MACD MOMENTUM IN TREND DIRECTION ==========
        # LONG: 1d trend up + 4h trend up + 1h Supertrend up + MACD histogram expanding positive
        long_condition = (
            trend_1d == 1 and
            trend_4h == 1 and
            st_trend_val == 1 and
            hist_val > 0 and
            hist_val > hist_prev and  # Histogram expanding
            price > kama_val  # Price above KAMA
        )
        
        # SHORT: 1d trend down + 4h trend down + 1h Supertrend down + MACD histogram expanding negative
        short_condition = (
            trend_1d == -1 and
            trend_4h == -1 and
            st_trend_val == -1 and
            hist_val < 0 and
            hist_val < hist_prev and  # Histogram expanding (more negative)
            price < kama_val  # Price below KAMA
        )
        
        # Determine position size based on conviction and volatility
        # Dynamic sizing: reduce size when ATR is high relative to price
        atr_pct = atr / price if price > 0 else 0.02
        vol_adjustment = min(1.5, TARGET_ATR_PCT / atr_pct) if atr_pct > 0 else 1.0
        
        # High conviction: all three timeframes align
        high_conviction_long = long_condition and st_trend_4h_val == 1 and st_trend_1d_val == 1
        high_conviction_short = short_condition and st_trend_4h_val == -1 and st_trend_1d_val == -1
        
        if long_condition:
            base_size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            size = min(SIZE_HIGH, base_size * vol_adjustment)  # Cap at max
            size = max(SIZE_BASE * 0.5, size)  # Floor at half base
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            base_size = SIZE_HIGH if high_conviction_short else SIZE_BASE
            size = min(SIZE_HIGH, base_size * vol_adjustment)  # Cap at max
            size = max(SIZE_BASE * 0.5, size)  # Floor at half base
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals