#!/usr/bin/env python3
"""
EXPERIMENT #029 - HMA RSI Z-Score Pullback 1h+4h MTF
==================================================================================================
Hypothesis: Return to 1h primary timeframe (proven in #027 with Sharpe=0.426) but add Z-score
regime filter from #019 (Sharpe=0.332). This combines the best elements of successful strategies:
1h+4h MTF structure, HMA trend, RSI pullback, plus Z-score for regime detection.

Key improvements vs #028 (Sharpe=0.035):
1. 1h PRIMARY (not 30m): Cleaner signals, less noise, proven in #027
2. HMA instead of KAMA: HMA has better lag reduction (proven in best strategy Sharpe=0.537)
3. Z-score regime filter: Avoid entries in extreme overbought/oversold regimes
4. 2.0*ATR stoploss (not 1.5*ATR): Less premature stopouts, more room for trades to work
5. Higher base size (0.25 vs 0.20): Better capital utilization while staying conservative

Why this should beat Sharpe=0.537:
- 1h timeframe captures more intraday opportunities than 4h
- HMA provides cleaner trend signals than KAMA
- Z-score filter avoids entering at extremes (reduces drawdown)
- 4h trend filter maintains directional bias
- Proper risk management with ATR stops and take-profit scaling
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_zscore_pullback_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.zeros(len(series))
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i - span + 1:i + 1] * weights)
        return result
    
    close_series = np.array(close)
    wma_half = wma(close_series, half)
    wma_full = wma(close_series, period)
    
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    # Pad beginning with zeros
    result = np.zeros(n)
    start_idx = period
    if start_idx < n:
        result[start_idx:] = hma[start_idx:]
    
    return result


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


def calculate_zscore(close, period=20):
    """Calculate Z-score for regime detection"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    for i in range(period, n):
        mean = np.mean(close[i - period:i])
        std = np.std(close[i - period:i])
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    zscore_1h = calculate_zscore(close, period=20)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(close_4h, period=21)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        # Align to 1h timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE with discrete levels
    SIZE_BASE = 0.25    # Base position (25%)
    SIZE_HIGH = 0.35    # High conviction (35%)
    SIZE_HALF = 0.15    # Half position for take profit
    
    # ATR stoploss - STANDARD (2.0*ATR)
    ATR_STOP_MULT = 2.0
    
    # RSI pullback zones
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    
    # Z-score regime filter
    ZSCORE_MAX = 1.5    # Don't enter if Z-score > 1.5 (overbought)
    ZSCORE_MIN = -1.5   # Don't enter if Z-score < -1.5 (oversold)
    
    first_valid = 100
    
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
        hma_val = hma_1h[i]
        zscore_val = zscore_1h[i]
        vol = volume[i]
        
        # 4h trend filters (MASTER FILTER)
        hma_4h_val = hma_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if hma_4h_val > 0:
            if price > hma_4h_val:
                trend_4h = 1
            elif price < hma_4h_val:
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
                    signals[i] = SIZE_HALF  # Reduce to half
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
                    signals[i] = -SIZE_HALF  # Reduce to half
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
        # LONG: 4h trend up + 1h price above HMA + RSI pullback (40-55) + Z-score not extreme
        long_condition = (
            trend_4h == 1 and
            hma_val > 0 and price > hma_val and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            zscore_val <= ZSCORE_MAX
        )
        
        # SHORT: 4h trend down + 1h price below HMA + RSI pullback (45-60) + Z-score not extreme
        short_condition = (
            trend_4h == -1 and
            hma_val > 0 and price < hma_val and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            zscore_val >= ZSCORE_MIN
        )
        
        # Determine position size based on conviction
        # High conviction: strong 4h trend + RSI in optimal zone
        high_conviction_long = long_condition and rsi_val >= 45 and rsi_val <= 50
        high_conviction_short = short_condition and rsi_val >= 50 and rsi_val <= 55
        
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