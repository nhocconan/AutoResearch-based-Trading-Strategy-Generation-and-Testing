#!/usr/bin/env python3
"""
EXPERIMENT #006 - KAMA MACD ADX Multi-Timeframe (1h Primary + 4h Trend)
==================================================================================================
Hypothesis: Current best (Sharpe=0.537) uses 4h+1d with HMA+RSI. This uses 1h+4h with KAMA+MACD+ADX.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - faster in trends, slower in noise.
MACD histogram provides momentum entry signals (different from RSI pullback).
ADX filter ensures we only trade when trend strength is sufficient (>25), avoiding choppy whipsaws.

Key innovations:
1. 1h PRIMARY + 4h HTF: More trade opportunities than 4h primary, cleaner than 30m
2. KAMA for trend: Adaptive to volatility, proven in quantitative literature
3. MACD histogram entry: Momentum-based, catches moves earlier than RSI pullback
4. ADX strength filter: Only trade when ADX > 25 (strong trend), skip ranging markets
5. Conservative sizing: 0.20-0.30 discrete levels, 2.0 ATR stoploss with trail

Why this should beat #005 (Sharpe=0.537):
- 1h timeframe captures more moves than 4h while avoiding 30m noise
- KAMA adapts better than HMA in mixed trending/ranging conditions
- MACD histogram gives earlier entries than waiting for RSI pullback
- ADX filter eliminates low-quality trades in choppy markets (major DD source)
- Different signal combo = uncorrelated returns, potential diversification benefit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_macd_adx_mtf_1h_4h_v1"
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
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market efficiency ratio (ER)
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast SC - slow SC) + slow SC)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """
    Calculate MACD line, signal line, and histogram
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal
    """
    n = len(close)
    if n < slow + signal_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    histogram = macd_line - signal_line
    
    return macd_line.values, signal_line.values, histogram.values


def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = weak/ranging
    """
    n = len(close)
    if n < period * 3:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initialize sums
    plus_sum = np.sum(plus_dm[1:period + 1])
    minus_sum = np.sum(minus_dm[1:period + 1])
    tr_sum = np.sum(tr[1:period + 1])
    
    for i in range(period, n):
        if i == period:
            plus_sum = np.sum(plus_dm[1:period + 1])
            minus_sum = np.sum(minus_dm[1:period + 1])
            tr_sum = np.sum(tr[1:period + 1])
        else:
            plus_sum = plus_sum - plus_dm[i - 1] + plus_dm[i]
            minus_sum = minus_sum - minus_dm[i - 1] + minus_dm[i]
            tr_sum = tr_sum - tr[i - 1] + tr[i]
        
        if tr_sum > 0:
            plus_di[i] = 100 * plus_sum / tr_sum
            minus_di[i] = 100 * minus_sum / tr_sum
        
        if (plus_di[i] + minus_di[i]) > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_1h_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    macd_1h, signal_1h, hist_1h = calculate_macd(close, fast=12, slow=26, signal_period=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
        adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.20   # Base position
    SIZE_HIGH = 0.30   # High conviction
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # ADX threshold for trend strength
    ADX_THRESHOLD = 25
    
    # MACD histogram threshold for entry
    MACD_THRESHOLD = 0
    
    first_valid = max(100, 60)
    
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
        kama_fast_val = kama_1h_fast[i]
        macd_hist = hist_1h[i]
        adx_val = adx_1h[i]
        
        # 4h trend filters (MASTER FILTER)
        kama_4h_val = kama_4h_aligned[i]
        adx_4h_val = adx_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if kama_4h_val > 0 and price > kama_4h_val:
            trend_4h = 1
        elif kama_4h_val > 0 and price < kama_4h_val:
            trend_4h = -1
        
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
        
        # ========== ENTRY LOGIC - MACD MOMENTUM IN TREND DIRECTION ==========
        # Check ADX for trend strength (both 1h and 4h)
        trend_strong_1h = adx_val > ADX_THRESHOLD
        trend_strong_4h = adx_4h_val > ADX_THRESHOLD if adx_4h_val > 0 else False
        
        # LONG: 4h trend up + 1h ADX strong + MACD histogram crossing up + price above KAMA
        long_condition = (
            trend_4h == 1 and
            trend_strong_1h and
            macd_hist > MACD_THRESHOLD and
            macd_hist > hist_1h[i - 1] if i > 0 else True and
            price > kama_val and
            kama_fast_val > kama_val
        )
        
        # SHORT: 4h trend down + 1h ADX strong + MACD histogram crossing down + price below KAMA
        short_condition = (
            trend_4h == -1 and
            trend_strong_1h and
            macd_hist < MACD_THRESHOLD and
            macd_hist < hist_1h[i - 1] if i > 0 else True and
            price < kama_val and
            kama_fast_val < kama_val
        )
        
        # High conviction: 4h ADX also strong
        high_conviction_long = long_condition and trend_strong_4h
        high_conviction_short = short_condition and trend_strong_4h
        
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