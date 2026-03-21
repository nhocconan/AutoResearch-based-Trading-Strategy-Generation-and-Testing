#!/usr/bin/env python3
"""
EXPERIMENT #009 - Bollinger Squeeze + MACD Momentum with Daily Trend (6h Primary)
==================================================================================================
Hypothesis: Current best uses 4h+1d with HMA+RSI. This tests 6h+1d with BB squeeze + MACD momentum.
6h timeframe should have fewer false signals than 4h, cleaner trends than 1h/30m.
Bollinger Band squeeze detects low volatility periods before breakouts.
MACD histogram confirms momentum direction. Daily trend filter prevents counter-trend trades.

Key innovations:
1. 6h PRIMARY + 1d HTF: Even fewer trades than 4h, less noise, better risk/reward
2. BB Squeeze: BW percentile < 20% indicates compression before expansion
3. MACD histogram: Confirms momentum direction (not just crossover)
4. Daily SMA(50) trend: Simple but effective master filter
5. Conservative sizing: 0.20-0.30 discrete levels, 2.5 ATR stoploss (wider for 6h)

Why this should beat #005 (Sharpe=0.537):
- 6h has less noise than 4h, fewer whipsaws
- BB squeeze captures volatility expansion (proven breakout signal)
- MACD histogram more responsive than RSI for momentum
- Daily SMA(50) is classic trend filter used by institutions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "bb_squeeze_macd_daily_trend_6h_v1"
timeframe = "6h"
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


def calculate_sma(series, period):
    """Calculate Simple Moving Average"""
    n = len(series)
    if n < period:
        return np.zeros(n)
    
    sma = np.zeros(n)
    for i in range(period - 1, n):
        sma[i] = np.mean(series[i - period + 1:i + 1])
    
    return sma


def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    n = len(series)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    ema[period - 1] = np.mean(series[:period])
    
    multiplier = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = (series[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """
    Calculate Bollinger Bands
    Returns: upper, middle, lower, bandwidth, percent_b
    """
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = calculate_sma(close, period)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    percent_b = np.zeros(n)
    
    for i in range(period - 1, n):
        if middle[i] == 0:
            continue
        
        window = close[i - period + 1:i + 1]
        std = np.std(window)
        
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        bandwidth[i] = (upper[i] - lower[i]) / middle[i]
        
        if upper[i] != lower[i]:
            percent_b[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
        else:
            percent_b[i] = 0.5
    
    return upper, middle, lower, bandwidth, percent_b


def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator
    Returns: macd_line, signal_line, histogram
    """
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line is EMA of MACD
    signal_line = np.zeros(n)
    first_valid = slow - 1
    if first_valid + signal <= n:
        macd_for_signal = macd_line[first_valid:first_valid + signal]
        signal_line[first_valid + signal - 1] = np.mean(macd_for_signal)
        
        multiplier = 2 / (signal + 1)
        for i in range(first_valid + signal, n):
            signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bb_squeeze_percentile(bandwidth, lookback=100):
    """
    Calculate Bollinger Bandwidth percentile over lookback period
    Low percentile = squeeze (compression)
    """
    n = len(bandwidth)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        if bandwidth[i] == 0:
            percentile[i] = 50
            continue
        
        window = bandwidth[i - lookback + 1:i + 1]
        valid_window = window[window > 0]
        
        if len(valid_window) == 0:
            percentile[i] = 50
            continue
        
        rank = np.sum(valid_window <= bandwidth[i])
        percentile[i] = (rank / len(valid_window)) * 100
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 6h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_6h = calculate_atr(high, low, close, period=14)
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_pct_b = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_squeeze_pct = calculate_bb_squeeze_percentile(bb_bandwidth, lookback=100)
    
    # MACD
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Trend EMAs
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # ========== 1d INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Daily SMA(50) for trend direction
        sma_50_1d = calculate_sma(close_1d, 50)
        sma_20_1d = calculate_sma(close_1d, 20)
        atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
        
        # Align to 6h timeframe (auto shift for completed bars)
        sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
        sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
        
    except Exception:
        sma_50_1d_aligned = np.zeros(n)
        sma_20_1d_aligned = np.zeros(n)
        atr_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.20   # Base position
    SIZE_HIGH = 0.30   # High conviction
    
    # ATR stoploss (wider for 6h timeframe)
    ATR_STOP_MULT = 2.5
    
    # BB squeeze threshold (low bandwidth = compression)
    BB_SQUEEZE_THRESHOLD = 25  # Bottom 25% of bandwidth
    
    # MACD momentum threshold
    MACD_MOMENTUM_THRESHOLD = 0.0
    
    first_valid = max(150, 100)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_6h[i]) or atr_6h[i] == 0 or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_6h[i]
        
        # Bollinger signals
        bb_squeeze = bb_squeeze_pct[i] < BB_SQUEEZE_THRESHOLD
        bb_expansion = bb_bandwidth[i] > bb_bandwidth[i - 1] if i > 0 else False
        
        # MACD signals
        macd_momentum = macd_hist[i]
        macd_cross_up = macd_hist[i] > 0 and macd_hist[i - 1] <= 0 if i > 0 else False
        macd_cross_down = macd_hist[i] < 0 and macd_hist[i - 1] >= 0 if i > 0 else False
        
        # EMA trend
        ema_trend = 0
        if ema_21[i] > ema_50[i]:
            ema_trend = 1
        elif ema_21[i] < ema_50[i]:
            ema_trend = -1
        
        # 1d trend filters (MASTER FILTER)
        sma_50_1d_val = sma_50_1d_aligned[i]
        sma_20_1d_val = sma_20_1d_aligned[i]
        
        # Determine daily trend direction
        daily_trend = 0
        if sma_50_1d_val > 0 and price > sma_50_1d_val:
            daily_trend = 1
        elif sma_50_1d_val > 0 and price < sma_50_1d_val:
            daily_trend = -1
        
        # SMA20 vs SMA50 confirmation
        if sma_20_1d_val > sma_50_1d_val and sma_50_1d_val > 0:
            daily_trend = max(daily_trend, 1)
        elif sma_20_1d_val < sma_50_1d_val and sma_50_1d_val > 0:
            daily_trend = min(daily_trend, -1)
        
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
            
            # Stoploss check (2.5*ATR for 6h)
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
                    signals[i] = SIZE_BASE  # Reduce to half
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
                    signals[i] = -SIZE_BASE  # Reduce to half
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
        
        # ========== ENTRY LOGIC - BB SQUEEZE BREAKOUT + MACD MOMENTUM ==========
        # LONG: Daily trend up + BB squeeze + MACD momentum up + price above BB middle
        long_condition = (
            daily_trend == 1 and
            bb_squeeze and
            macd_momentum > MACD_MOMENTUM_THRESHOLD and
            price > bb_middle[i] and
            ema_trend == 1
        )
        
        # SHORT: Daily trend down + BB squeeze + MACD momentum down + price below BB middle
        short_condition = (
            daily_trend == -1 and
            bb_squeeze and
            macd_momentum < MACD_MOMENTUM_THRESHOLD and
            price < bb_middle[i] and
            ema_trend == -1
        )
        
        # Determine position size based on conviction
        # High conviction: strong daily trend + MACD cross
        high_conviction_long = long_condition and macd_cross_up and daily_trend == 1
        high_conviction_short = short_condition and macd_cross_down and daily_trend == -1
        
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