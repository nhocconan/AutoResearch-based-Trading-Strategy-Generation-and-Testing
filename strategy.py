#!/usr/bin/env python3
"""
EXPERIMENT #008 - MTF KAMA Trend + RSI Pullback + Z-score (1h+4h v1)
==================================================================================================
Hypothesis: Switch primary timeframe to 1h (vs 15m in current best) for cleaner signals with less
noise. Use 4h KAMA (Kaufman Adaptive Moving Average) for trend - KAMA adapts to volatility better
than HMA/EMA. Add Z-score filter to avoid entering at extreme deviations. Simpler 2-TF approach
vs failed 3-TF strategies (#001-#007 all failed).

Why this should work:
- 1h primary = fewer false signals than 15m, more trades than 4h
- KAMA adapts to market regime (fast in trends, slow in chop) - proven in literature
- Z-score(20) filter avoids buying tops/selling bottoms
- 4h trend filter prevents counter-trend trades (learned from #001-#007 failures)
- Conservative signal size 0.30 with proper stoploss/TP management
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_zscore_1h_4h_v1"
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


def calculate_kama(close, er_period=10, fast_sc=2/11, slow_sc=2/31):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility - fast during trends, slow during chop
    """
    n = len(close)
    if n < er_period:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    sc = np.zeros(n)
    for i in range(n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period - 1] = close[er_period - 1]
    
    for i in range(er_period, n):
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


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(n):
        if std[i] > 0:
            zscore[i] = (close[i] - mean[i]) / std[i]
        else:
            zscore[i] = 0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    kama_1h = calculate_kama(close, er_period=10)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(c_4h, er_period=10)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        c_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
        c_4h_aligned = np.zeros(n)
    
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
    
    # Z-score filter - avoid extreme deviations
    ZSCORE_MAX = 1.5
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(50, 14 * 2, 20, 10)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(zscore_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            if i > 0:
                position_side[i] = position_side[i - 1]
            continue
        
        # Get aligned 4h values
        kama_4h_val = kama_4h_aligned[i] if i < len(kama_4h_aligned) else 0
        close_4h_val = c_4h_aligned[i] if i < len(c_4h_aligned) else close[i]
        
        # Determine 4h trend direction (price vs KAMA)
        trend_4h = 0
        if kama_4h_val > 0 and close_4h_val > 0:
            if close_4h_val > kama_4h_val:
                trend_4h = 1  # Bullish
            elif close_4h_val < kama_4h_val:
                trend_4h = -1  # Bearish
        
        # Z-score filter - avoid entering at extremes
        if abs(zscore_1h[i]) > ZSCORE_MAX:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                # Close position if we're at extreme deviation
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
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
            
            # Update highest/lowest since entry
            current_high = max(prev_high, close[i])
            current_low = min(prev_low, close[i])
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            price = close[i]
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Check if trend reversed - close position
            if prev_side == 1 and trend_4h == -1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            elif prev_side == -1 and trend_4h == 1:
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
        
        # Entry logic: 4h trend + 1h RSI pullback + Z-score filter
        price = close[i]
        
        if trend_4h == 1:  # Bullish trend on 4h
            # RSI pullback on 1h (not overbought)
            # Z-score not at extreme high
            if (RSI_LONG_MIN <= rsi_1h[i] <= RSI_LONG_MAX and 
                zscore_1h[i] < ZSCORE_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1:  # Bearish trend on 4h
            # RSI pullback on 1h (not oversold)
            # Z-score not at extreme low
            if (RSI_SHORT_MIN <= rsi_1h[i] <= RSI_SHORT_MAX and 
                zscore_1h[i] > -ZSCORE_MAX):
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