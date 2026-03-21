#!/usr/bin/env python3
"""
EXPERIMENT #039 - KAMA Donchian Trend Pullback (4h Primary + 1d HTF)
==================================================================================================
Hypothesis: Current best uses daily trend with 4h entries. Testing 4h primary (cleaner than 15m/30m)
with 1d HTF trend filter (proven in best strategy). Using KAMA (adaptive) instead of HMA for better
whipsaw protection, and Donchian Channel for trend confirmation. Simpler RSI zones (30-70) to
capture more pullback opportunities without over-filtering.

Key innovations vs #038:
1. 4h PRIMARY + 1d HTF: Cleaner signals than 15m, matches best strategy's TF combo
2. KAMA instead of HMA: Adaptive to volatility, proven in #030/#035 (Sharpe 0.4+)
3. Donchian Channel trend: Simple breakout confirmation, less lag than HMA crossover
4. Simpler RSI zones: 30-70 instead of 40-60 (more entry opportunities)
5. Discrete position sizing: Only 0.0, ±0.25, ±0.35 (reduce fee churn)
6. Standard ATR stoploss: 2.0*ATR instead of 1.8*ATR (less premature exits)
7. Bollinger regime filter: Only trade when BW > median (avoid low-vol chop)

Why this should beat Sharpe=0.537:
- 4h captures swing moves without 15m noise
- 1d trend filter is strongest (best strategy used this)
- KAMA adapts to volatility better than fixed HMA
- Donchian adds independent trend confirmation
- Discrete sizing reduces fee drag from constant adjustments
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_donchian_rsi_pullback_4h_1d_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts to market volatility - moves fast in trends, slow in chop
    """
    n = len(close)
    if n < slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # First KAMA value = close price at er_period
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - tracks highest high and lowest low over period
    Returns: upper_band, lower_band, middle_band
    """
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2
    
    return upper, lower, middle


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """
    Bollinger Bands with Bandwidth for regime detection
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    # Handle NaN/inf
    upper = np.nan_to_num(upper, nan=0.0)
    lower = np.nan_to_num(lower, nan=0.0)
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    
    return upper, lower, sma, bandwidth


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 4h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    kama_4h_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=21)
    kama_4h_slow = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    bb_upper, bb_lower, bb_mid, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate median bandwidth for regime filter
    valid_bw = bb_bandwidth[bb_bandwidth > 0]
    if len(valid_bw) > 0:
        bw_median = np.median(valid_bw)
    else:
        bw_median = 0.001
    
    # ========== 1d INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # 1d KAMA for trend direction
        kama_1d = calculate_kama(close_1d, er_period=10, fast_period=2, slow_period=21)
        atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
        donchian_1d_upper, donchian_1d_lower, donchian_1d_mid = calculate_donchian(high_1d, low_1d, period=20)
        
        # Align to 4h timeframe (auto shift for completed bars)
        kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
        donchian_1d_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_1d_mid)
        
    except Exception:
        kama_1d_aligned = np.zeros(n)
        donchian_1d_mid_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE LEVELS ONLY
    SIZE_BASE = 0.25   # Base position (25% of capital)
    SIZE_HIGH = 0.35   # High conviction (35% of capital)
    
    # ATR stoploss - standard
    ATR_STOP_MULT = 2.0
    
    # RSI pullback zones - simple and wide
    RSI_LONG_MIN = 30
    RSI_LONG_MAX = 70
    RSI_SHORT_MIN = 30
    RSI_SHORT_MAX = 70
    
    # Regime filter - only trade when volatility > median
    REGIME_FILTER = True
    
    first_valid = 200
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_4h[i]) or atr_4h[i] == 0 or np.isnan(rsi_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        rsi_val = rsi_4h[i]
        kama_fast_val = kama_4h_fast[i]
        kama_slow_val = kama_4h_slow[i]
        bw_val = bb_bandwidth[i]
        
        # 1d trend filters (MASTER FILTER)
        kama_1d_val = kama_1d_aligned[i]
        donchian_1d_mid_val = donchian_1d_mid_aligned[i]
        
        # Determine 1d trend direction
        trend_1d = 0
        if kama_1d_val > 0:
            if price > kama_1d_val:
                trend_1d = 1
            elif price < kama_1d_val:
                trend_1d = -1
        
        # Donchian confirmation on 1d
        if donchian_1d_mid_val > 0:
            if price > donchian_1d_mid_val and trend_1d == 0:
                trend_1d = 1
            elif price < donchian_1d_mid_val and trend_1d == 0:
                trend_1d = -1
        
        # Regime filter - skip if bandwidth too low (choppy market)
        if REGIME_FILTER and bw_val < bw_median * 0.8:
            if position_side[i - 1] != 0:
                # Close existing position in low vol regime
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
            continue
        
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
            
            # Stoploss check (2.0*ATR)
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
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE / 2  # Reduce to half (0.125)
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
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
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE / 2  # Reduce to half (-0.125)
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
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
        
        # ========== ENTRY LOGIC - RSI PULLBACK IN TREND DIRECTION ==========
        # 4h KAMA trend
        kama_trend_4h = 0
        if kama_fast_val > kama_slow_val:
            kama_trend_4h = 1
        elif kama_fast_val < kama_slow_val:
            kama_trend_4h = -1
        
        # Donchian breakout confirmation
        donchian_trend_4h = 0
        if price > donchian_mid[i] and donchian_mid[i] > 0:
            donchian_trend_4h = 1
        elif price < donchian_mid[i] and donchian_mid[i] > 0:
            donchian_trend_4h = -1
        
        # LONG: 1d trend up + 4h KAMA up + 4h Donchian up + RSI pullback (30-70)
        long_condition = (
            trend_1d == 1 and
            kama_trend_4h == 1 and
            donchian_trend_4h == 1 and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX
        )
        
        # SHORT: 1d trend down + 4h KAMA down + 4h Donchian down + RSI pullback (30-70)
        short_condition = (
            trend_1d == -1 and
            kama_trend_4h == -1 and
            donchian_trend_4h == -1 and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX
        )
        
        # Determine position size based on conviction
        # High conviction: RSI in optimal zone (40-60 for long, 40-60 for short)
        high_conviction_long = long_condition and rsi_val >= 40 and rsi_val <= 60
        high_conviction_short = short_condition and rsi_val >= 40 and rsi_val <= 60
        
        if long_condition:
            size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            size = SIZE_HIGH if high_conviction_short else SIZE_BASE
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