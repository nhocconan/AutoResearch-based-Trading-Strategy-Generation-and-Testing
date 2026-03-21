#!/usr/bin/env python3
"""
EXPERIMENT #036 - HMA RSI Pullback with 1h Primary + 4h Trend Filter (v1)
==================================================================================================
Hypothesis: Current best (Sharpe=0.537) uses 4h+1d. This tests 1h+4h combination which should
produce MORE trades while maintaining clean trend signals. 1h timeframe captures more pullback
opportunities than 4h, while 4h filter still provides strong trend direction.

Key changes from current best (#035):
1. 1h PRIMARY (not 4h): More trade opportunities, captures intraday pullbacks better
2. 4h HTF trend filter (not 1d): Proven in experiments #027, #030, #034 (Sharpe 0.41-0.43)
3. HMA instead of KAMA: Current best uses HMA successfully, KAMA underperformed in #035
4. RSI zones: 40-60 (proven in current best) instead of 35-65 (#035)
5. Stoploss: 2.0*ATR (current best) instead of 1.5*ATR (#035 was too tight)
6. Position sizing: 0.20/0.30 discrete levels (conservative, reduces fee churn)
7. Volume confirmation: Require above-average volume on entry bars

Why this should beat Sharpe=0.537:
- 1h timeframe = 4x more bars than 4h = more entry opportunities
- 4h trend filter is strong enough (proven in multiple experiments)
- HMA is faster than KAMA at detecting trend changes
- 2.0*ATR stoploss gives trades room to breathe (1.5*ATR was too tight in #035)
- Volume filter reduces false breakouts
- Same proven MTF structure that worked in experiments #027, #030, #034
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_volume_pullback_1h_4h_v1"
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


def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close = np.array(close)
    
    # WMA helper
    def wma(data, span):
        result = np.zeros(len(data))
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            result[i] = np.sum(data[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma


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


def calculate_sma(data, period):
    """Simple Moving Average"""
    n = len(data)
    if n < period:
        return np.zeros(n)
    
    sma = np.zeros(n)
    for i in range(period - 1, n):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h_fast = calculate_hma(close, period=8)
    hma_1h_slow = calculate_hma(close, period=21)
    hma_1h_trend = calculate_hma(close, period=48)
    vol_sma_1h = calculate_sma(volume, period=20)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(close_4h, period=21)
        hma_4h_fast = calculate_hma(close_4h, period=8)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        hma_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_fast)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
        hma_4h_fast_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to minimize fee churn
    SIZE_BASE = 0.20   # Base position (20% of capital)
    SIZE_HIGH = 0.30   # High conviction (30% of capital)
    
    # ATR stoploss - 2.0*ATR (current best, not too tight)
    ATR_STOP_MULT = 2.0
    
    # RSI pullback zones - proven in current best
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi_val = rsi_1h[i]
        hma_fast = hma_1h_fast[i]
        hma_slow = hma_1h_slow[i]
        hma_trend = hma_1h_trend[i]
        vol = volume[i]
        vol_avg = vol_sma_1h[i]
        
        # 4h trend filters (MASTER FILTER)
        hma_4h_val = hma_4h_aligned[i]
        hma_4h_fast_val = hma_4h_fast_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if hma_4h_val > 0 and price > hma_4h_val:
            trend_4h = 1
        elif hma_4h_val > 0 and price < hma_4h_val:
            trend_4h = -1
        
        # Confirm with fast HMA
        if hma_4h_fast_val > hma_4h_val:
            trend_4h = max(trend_4h, 1)
        elif hma_4h_fast_val < hma_4h_val:
            trend_4h = min(trend_4h, -1)
        
        # Volume confirmation
        vol_above_avg = vol > vol_avg * 1.0 if vol_avg > 0 else False
        
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
                    signals[i] = SIZE_BASE / 2  # Reduce to half
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
                    signals[i] = -SIZE_BASE / 2  # Reduce to half
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
        # LONG: 4h trend up + 1h HMA aligned + RSI pullback (40-60) + volume confirmation
        long_condition = (
            trend_4h == 1 and
            hma_fast > hma_slow and
            hma_slow > hma_trend and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            vol_above_avg
        )
        
        # SHORT: 4h trend down + 1h HMA aligned + RSI pullback (40-60) + volume confirmation
        short_condition = (
            trend_4h == -1 and
            hma_fast < hma_slow and
            hma_slow < hma_trend and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            vol_above_avg
        )
        
        # Determine position size based on conviction
        # High conviction: strong 4h trend (fast > slow HMA)
        high_conviction_long = long_condition and hma_4h_fast_val > hma_4h_val
        high_conviction_short = short_condition and hma_4h_fast_val < hma_4h_val
        
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