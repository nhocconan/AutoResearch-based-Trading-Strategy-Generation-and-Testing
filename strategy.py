#!/usr/bin/env python3
"""
EXPERIMENT #086 - ADAPTIVE_HMA_RSI_ATR_1H_V1
==================================================================================================
Hypothesis: Simpler 1h ensemble with adaptive ATR-based position sizing.
- Core: HMA(21) trend + RSI(14) momentum + ATR(14) volatility filter
- Position sizing: Scale by inverse ATR (lower vol = larger position)
- Regime: BBW percentile determines entry thresholds
- Discrete signals: 0.0, ±0.20, ±0.35 to reduce churn costs
- Stoploss: 2.0*ATR trailing, take profit at 2R then trail
- Timeframe: 1h (balances noise vs trade frequency)
- Max position: 0.35 (conservative for drawdown control)

Why this should work:
- 1h has proven stable in experiments #077, #082, #083
- ATR-based sizing adapts to volatility automatically
- Simpler logic = fewer bugs than complex ensemble voting
- Discrete levels reduce fee churn from signal changes
"""

import numpy as np
import pandas as pd

name = "adaptive_hma_rsi_atr_1h_v1"
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
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        current = bbw[i]
        percentile[i] = np.sum(window <= current) / len(window)
    
    return percentile


def calculate_sma(close, period=200):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = np.zeros(n)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Calculate all indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    hma = calculate_hma(close, period=21)
    sma200 = calculate_sma(close, period=200)
    bb_upper, bb_middle, bb_lower, bbw = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct = calculate_bbw_percentile(bbw, lookback=100)
    
    # Position sizing parameters
    BASE_SIZE = 0.275
    SIZE_LOW = 0.20
    SIZE_MED = 0.275
    SIZE_HIGH = 0.35
    ATR_STOP_MULT = 2.0
    
    # ATR-based position sizing (inverse volatility)
    # Target: risk 5% per trade, so size = 0.05 / (ATR_mult * ATR_pct)
    atr_pct = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 0:
            atr_pct[i] = atr[i] / close[i-1]
        else:
            atr_pct[i] = 0.01
    
    # Regime thresholds
    LOW_VOL_PCT = 0.30
    HIGH_VOL_PCT = 0.70
    
    # Signal thresholds
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    
    first_valid = max(200, 14 * 2, 20, 100)
    
    # Track position state - use lists to avoid read-only issues
    signals = np.zeros(n)
    position_side = np.zeros(n, dtype=np.int32)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=np.int32)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr[i]) or atr[i] == 0 or np.isnan(bbw_pct[i]) or np.isnan(sma200[i]):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        regime = bbw_pct[i]
        current_atr_pct = atr_pct[i]
        
        # Calculate adaptive position size based on ATR
        # Lower ATR = larger position (inverse volatility scaling)
        target_risk = 0.05  # 5% risk per trade
        if current_atr_pct > 0:
            raw_size = target_risk / (ATR_STOP_MULT * current_atr_pct)
            # Cap at max position size
            adaptive_size = min(raw_size, SIZE_HIGH)
            adaptive_size = max(adaptive_size, SIZE_LOW)
        else:
            adaptive_size = SIZE_MED
        
        # Check existing position for stoploss/takeprofit
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
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = adaptive_size * 0.5  # Reduce to half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            else:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -adaptive_size * 0.5  # Reduce to half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Generate signals
        # Trend filter: price vs HMA and SMA200
        trend_long = price > hma[i] and price > sma200[i]
        trend_short = price < hma[i] and price < sma200[i]
        
        # Momentum filter: RSI
        momentum_long = RSI_LONG_MIN <= rsi[i] <= RSI_LONG_MAX
        momentum_short = RSI_SHORT_MIN <= rsi[i] <= RSI_SHORT_MAX
        
        # Regime-adaptive entry thresholds
        if regime < LOW_VOL_PCT:
            # Low volatility = trend regime (require stronger trend confirmation)
            entry_threshold = 0.7
        elif regime > HIGH_VOL_PCT:
            # High volatility = mean reversion regime (be more conservative)
            entry_threshold = 0.8
        else:
            # Normal volatility = standard thresholds
            entry_threshold = 0.75
        
        # Calculate signal strength
        signal_strength = 0.0
        
        if trend_long and momentum_long:
            signal_strength = 1.0
        elif trend_short and momentum_short:
            signal_strength = -1.0
        
        # Determine position based on signal strength and regime
        if signal_strength >= entry_threshold:
            signals[i] = adaptive_size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            
        elif signal_strength <= -entry_threshold:
            signals[i] = -adaptive_size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals