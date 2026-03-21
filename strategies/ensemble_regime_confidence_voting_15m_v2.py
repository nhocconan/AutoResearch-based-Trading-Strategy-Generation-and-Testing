#!/usr/bin/env python3
"""
EXPERIMENT #085 - ENSEMBLE_REGIME_CONFIDENCE_VOTING_15M_V2
==================================================================================================
Hypothesis: Combine 3 independent signal generators with regime-adaptive weighting.
- Signal 1: Trend (HMA + Supertrend agreement)
- Signal 2: Momentum (RSI + MACD histogram)
- Signal 3: Mean Reversion (Z-score + Bollinger position)
- Regime: BBW percentile determines which signals to trust more
- Confidence: More signals agree = larger position (0.20 to 0.35)
- Timeframe: 15m (proven optimal in experiments #031, #034, #035)
- Max position: 0.35 (conservative for drawdown control)
- Stoploss: 2.0*ATR with trailing at 1R profit

Why this should work:
- Ensemble reduces false signals from any single indicator
- Regime detection adapts to market conditions
- Confidence-based sizing rewards high-conviction setups
- Simpler than #040 but more robust across regimes
"""

import numpy as np
import pandas as pd

name = "ensemble_regime_confidence_voting_15m_v2"
timeframe = "15m"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    
    for i in range(1, n):
        ema_fast[i] = close[i] * (2.0 / (fast + 1)) + ema_fast[i - 1] * (1.0 - 2.0 / (fast + 1))
        ema_slow[i] = close[i] * (2.0 / (slow + 1)) + ema_slow[i - 1] * (1.0 - 2.0 / (slow + 1))
    
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    signal_line[slow - 1] = macd_line[slow - 1]
    for i in range(slow, n):
        signal_line[i] = macd_line[i] * (2.0 / (signal + 1)) + signal_line[i - 1] * (1.0 - 2.0 / (signal + 1))
    
    for i in range(slow - 1, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0
    
    return zscore


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
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Calculate all indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    zscore = calculate_zscore(close, period=20)
    hma = calculate_hma(close, period=21)
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper, bb_middle, bb_lower, bbw = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct = calculate_bbw_percentile(bbw, lookback=100)
    
    # Signal parameters
    SIZE_LOW = 0.20
    SIZE_MED = 0.275
    SIZE_HIGH = 0.35
    ATR_STOP_MULT = 2.0
    
    # Regime thresholds
    LOW_VOL_PCT = 0.30  # BBW percentile < 30% = low vol (trend regime)
    HIGH_VOL_PCT = 0.70  # BBW percentile > 70% = high vol (mean reversion regime)
    
    # Signal thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    ZSCORE_MAX = 2.0
    MACD_MIN = 0.0
    
    first_valid = max(200, 14 * 2, 20, 26 + 9, 100)
    
    # Track position state
    signals = np.zeros(n)
    position_side = np.zeros(n, dtype=np.int32)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=np.int32)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr[i]) or atr[i] == 0 or np.isnan(bbw_pct[i]):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        regime = bbw_pct[i]
        
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
                    signals[i] = SIZE_LOW
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
                    signals[i] = -SIZE_LOW
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
        
        # Generate ensemble signals
        trend_signal = 0
        momentum_signal = 0
        meanrev_signal = 0
        
        # Signal 1: Trend (HMA + Supertrend agreement)
        if price > hma[i] and st_direction[i] == 1:
            trend_signal = 1
        elif price < hma[i] and st_direction[i] == -1:
            trend_signal = -1
        
        # Signal 2: Momentum (RSI + MACD histogram)
        if RSI_LONG_MIN <= rsi[i] <= RSI_LONG_MAX and macd_hist[i] > MACD_MIN:
            momentum_signal = 1
        elif RSI_SHORT_MIN <= rsi[i] <= RSI_SHORT_MAX and macd_hist[i] < -MACD_MIN:
            momentum_signal = -1
        
        # Signal 3: Mean Reversion (Z-score + BB position)
        bb_position = (price - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 0 else 0.5
        if zscore[i] < -ZSCORE_MAX and bb_position < 0.3:
            meanrev_signal = 1
        elif zscore[i] > ZSCORE_MAX and bb_position > 0.7:
            meanrev_signal = -1
        
        # Regime-adaptive weighting
        if regime < LOW_VOL_PCT:
            # Low volatility = trend regime (weight trend higher)
            weights = [0.5, 0.3, 0.2]
        elif regime > HIGH_VOL_PCT:
            # High volatility = mean reversion regime (weight meanrev higher)
            weights = [0.2, 0.3, 0.5]
        else:
            # Normal volatility = balanced
            weights = [0.33, 0.33, 0.34]
        
        # Weighted vote
        vote_score = (trend_signal * weights[0] + 
                      momentum_signal * weights[1] + 
                      meanrev_signal * weights[2])
        
        # Count agreeing signals
        agree_count = 0
        if trend_signal > 0:
            agree_count += 1
        elif trend_signal < 0:
            agree_count -= 1
        
        if momentum_signal > 0:
            agree_count += 1
        elif momentum_signal < 0:
            agree_count -= 1
        
        if meanrev_signal > 0:
            agree_count += 1
        elif meanrev_signal < 0:
            agree_count -= 1
        
        # Determine position based on vote and confidence
        if vote_score > 0.3:
            if agree_count >= 2:
                signals[i] = SIZE_HIGH
                position_side[i] = 1
            elif agree_count == 1:
                signals[i] = SIZE_MED
                position_side[i] = 1
            else:
                signals[i] = SIZE_LOW
                position_side[i] = 1
            
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            
        elif vote_score < -0.3:
            if agree_count <= -2:
                signals[i] = -SIZE_HIGH
                position_side[i] = -1
            elif agree_count == -1:
                signals[i] = -SIZE_MED
                position_side[i] = -1
            else:
                signals[i] = -SIZE_LOW
                position_side[i] = -1
            
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals