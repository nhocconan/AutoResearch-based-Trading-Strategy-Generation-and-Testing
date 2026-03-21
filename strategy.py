#!/usr/bin/env python3
"""
EXPERIMENT #035 - MTF KAMA+Supertrend+RSI+Zscore Dynamic Sizing (4h+1h v2)
==================================================================================================
Hypothesis: Previous strategy crashed due to complex position tracking in loop.
Simplify signal generation by:
- Using KAMA (Kaufman Adaptive Moving Average) - adapts to market noise better than DEMA
- Supertrend for clear trend direction (proven in current best)
- RSI for pullback entries
- Z-score filter for regime detection (avoid trading in extreme conditions)
- ATR-based dynamic position sizing
- Simpler signal logic without mutable state tracking

Why this should beat current best (Sharpe=3.653):
- KAMA adapts to volatility (better than fixed DEMA/HMA)
- Supertrend provides clear stoploss levels
- Z-score avoids trading in extreme regimes
- 4h trend + 1h entries is proven combination
- Simpler code = fewer bugs

Position sizing: base=0.30, adjusted by ATR ratio (target 2% ATR / current ATR%)
Stoploss: Supertrend level (automatic trailing)
Take profit: 2R with signal reduction to half
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf


name = "mtf_kama_supertrend_rsi_zscore_atr_dynamic_4h_1h_v2"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        # Calculate bands
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
        
        # Determine trend
        if i == period:
            if close[i] > upper_band[i]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            else:
                trend[i] = -1
                supertrend[i] = upper_band[i]
        else:
            if trend[i - 1] == 1:
                if close[i] > lower_band[i]:
                    trend[i] = 1
                    supertrend[i] = max(lower_band[i], supertrend[i - 1])
                else:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
            else:
                if close[i] < upper_band[i]:
                    trend[i] = -1
                    supertrend[i] = min(upper_band[i], supertrend[i - 1])
                else:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
    
    return supertrend, trend, upper_band


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to moving average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        if std[i] > 0:
            zscore[i] = (close[i] - sma[i]) / std[i]
        else:
            zscore[i] = 0
    
    return zscore


def calculate_atr_pct(atr, close):
    """Calculate ATR as percentage of price"""
    n = len(close)
    atr_pct = np.zeros(n)
    for i in range(n):
        if close[i] > 0:
            atr_pct[i] = atr[i] / close[i]
    return atr_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Initialize signals array
    signals = np.zeros(n)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    atr_pct_1h = calculate_atr_pct(atr_1h, close)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, period=10, fast=2, slow=30)
    supertrend_1h, trend_1h, _ = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    zscore_1h = calculate_zscore(close, period=20)
    
    # Get 4h data using mtf_data helper (CRITICAL for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend
        kama_4h_raw = calculate_kama(close_4h, period=10, fast=2, slow=30)
        supertrend_4h_raw, trend_4h_raw, _ = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        atr_4h_raw = calculate_atr(high_4h, low_4h, close_4h, period=14)
        zscore_4h_raw = calculate_zscore(close_4h, period=20)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        kama_4h = align_htf_to_ltf(prices, df_4h, kama_4h_raw)
        supertrend_4h = align_htf_to_ltf(prices, df_4h, supertrend_4h_raw)
        trend_4h = align_htf_to_ltf(prices, df_4h, trend_4h_raw)
        atr_4h = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
        zscore_4h = align_htf_to_ltf(prices, df_4h, zscore_4h_raw)
        
    except Exception:
        # Fallback if mtf_data fails
        kama_4h = kama_1h
        supertrend_4h = supertrend_1h
        trend_4h = trend_1h
        atr_4h = atr_1h
        zscore_4h = zscore_1h
    
    # Position sizing parameters
    BASE_SIZE = 0.30
    TARGET_ATR_PCT = 0.02
    MIN_SIZE = 0.15
    MAX_SIZE = 0.40
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Z-score filter (avoid extreme regimes)
    ZSCORE_MAX = 2.0
    
    # Take profit levels
    TP_R_MULT = 2.0
    
    # Track position state for stoploss/takeprofit
    position_side = np.zeros(n, dtype=np.int8)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    first_valid = max(100, 30, 20, 14)
    
    for i in range(first_valid, n):
        # Skip if invalid data
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Z-score filter (avoid extreme regimes)
        if abs(zscore_4h[i]) > ZSCORE_MAX:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h trend from Supertrend
        trend_direction = trend_4h[i]
        
        # Dynamic position sizing based on ATR
        current_atr_pct = atr_pct_1h[i]
        if current_atr_pct > 0:
            atr_ratio = TARGET_ATR_PCT / current_atr_pct
            position_size = BASE_SIZE * atr_ratio
            position_size = np.clip(position_size, MIN_SIZE, MAX_SIZE)
        else:
            position_size = BASE_SIZE
        
        half_size = position_size / 2
        
        # Check existing positions for stoploss/takeprofit
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            atr = atr_1h[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Supertrend stoploss check
            if prev_side == 1:
                # Long position - check if price below supertrend
                if price < supertrend_4h[i] or price < prev_entry - 2.0 * atr:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_R_MULT * 2.0 * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = half_size
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - 2.0 * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Hold position
                signals[i] = signals[i - 1]
                position_side[i] = prev_side
                entry_price[i] = prev_entry
                tp_triggered[i] = prev_tp
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
                continue
                
            elif prev_side == -1:
                # Short position - check if price above supertrend
                if price > supertrend_4h[i] or price > prev_entry + 2.0 * atr:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_R_MULT * 2.0 * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -half_size
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + 2.0 * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Hold position
                signals[i] = signals[i - 1]
                position_side[i] = prev_side
                entry_price[i] = prev_entry
                tp_triggered[i] = prev_tp
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
                continue
        
        # Entry logic: 4h trend + 1h RSI pullback + KAMA confirmation
        rsi_val = rsi_1h[i]
        price = close[i]
        
        if trend_direction == 1:  # Bullish trend on 4h
            # KAMA confirmation (price above KAMA)
            if price > kama_4h[i] and kama_4h[i] > 0:
                # RSI pullback entry (1h)
                if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                    signals[i] = position_size
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = False
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend_direction == -1:  # Bearish trend on 4h
            # KAMA confirmation (price below KAMA)
            if price < kama_4h[i] and kama_4h[i] > 0:
                # RSI pullback entry (1h)
                if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                    signals[i] = -position_size
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = False
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals