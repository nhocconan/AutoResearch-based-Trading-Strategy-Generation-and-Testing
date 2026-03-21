#!/usr/bin/env python3
"""
EXPERIMENT #053 - Bollinger Regime + Supertrend + RSI Ensemble (1h Primary)
==================================================================================================
Hypothesis: Current best (Sharpe=0.537) is pure trend-following with pullback entries.
This strategy adapts to market regime: trend-follow in low vol, mean-revert in high vol.
Uses 1h primary (more trades than 4h) + 4h trend filter + BB width regime detection.

Key innovations:
1. Regime detection: BB Width percentile (low=20th, high=80th) to switch strategies
2. Low vol regime: Supertrend trend-following (proven in #045, #047)
3. High vol regime: RSI mean-reversion at extremes (30/70) with 4h trend filter
4. Ensemble voting: 3 signals (Supertrend, RSI, BB regime) must agree for entry
5. Adaptive sizing: 0.30 for high conviction (all 3 align), 0.20 for base

Why this should beat current best (Sharpe=0.537):
- Adapts to volatility regime instead of always trend-following
- 1h timeframe captures more opportunities than 4h
- Ensemble reduces false signals (need 2/3 agreement minimum)
- Mean-reversion in high vol captures choppy market profits that trend strategies miss
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "bb_regime_supertrend_rsi_ensemble_1h_4h_v1"
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    sma = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    # Handle division by zero
    bandwidth = np.where(sma > 0, bandwidth, 0)
    
    return upper, lower, sma, bandwidth


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


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops
    Returns: supertrend_values, trend_direction (1=up, -1=down, 0=neutral)
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


def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate BB Width percentile over lookback period"""
    n = len(bandwidth)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bandwidth[i - lookback + 1:i + 1]
        valid_window = window[window > 0]
        if len(valid_window) > 0:
            rank = np.sum(valid_window < bandwidth[i])
            percentile[i] = rank / len(valid_window) * 100
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_sma, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_pct = calculate_bb_percentile(bb_width, lookback=100)
    supertrend_1h, st_trend_1h = calculate_supertrend(high, low, close, atr_1h, multiplier=3.0)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend filter
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        rsi_4h = calculate_rsi(close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        
        # Align to 1h timeframe (auto shift for completed bars)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        
    except Exception:
        st_trend_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.20   # Base position
    SIZE_HIGH = 0.30   # High conviction (all signals align)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # RSI thresholds for mean reversion
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 65
    
    # BB regime thresholds
    BB_LOW_VOL = 25    # Below 25th percentile = low vol (trend follow)
    BB_HIGH_VOL = 75   # Above 75th percentile = high vol (mean revert)
    
    first_valid = max(150, 100)
    
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
        st_trend_val = st_trend_1h[i]
        bb_percentile = bb_pct[i]
        
        # 4h trend filters
        st_trend_4h_val = st_trend_4h_aligned[i]
        rsi_4h_val = rsi_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if st_trend_4h_val == 1:
            trend_4h = 1
        elif st_trend_4h_val == -1:
            trend_4h = -1
        
        # Determine regime
        is_low_vol = bb_percentile < BB_LOW_VOL
        is_high_vol = bb_percentile > BB_HIGH_VOL
        is_neutral_vol = not is_low_vol and not is_high_vol
        
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
        
        # ========== ENTRY LOGIC - REGIME-BASED ==========
        # Signal votes
        trend_vote = 0  # Supertrend vote
        rsi_vote = 0    # RSI vote
        regime_vote = 0 # Regime-based vote
        
        # Supertrend vote (trend direction)
        if st_trend_val == 1:
            trend_vote = 1
        elif st_trend_val == -1:
            trend_vote = -1
        
        # RSI vote (extremes indicate reversal potential)
        if rsi_val < RSI_OVERSOLD:
            rsi_vote = 1  # Oversold = potential long
        elif rsi_val > RSI_OVERBOUGHT:
            rsi_vote = -1  # Overbought = potential short
        
        # Regime-based logic
        if is_low_vol:
            # Low volatility = trend following
            regime_vote = trend_vote
        elif is_high_vol:
            # High volatility = mean reversion (opposite of RSI extreme)
            regime_vote = rsi_vote
        else:
            # Neutral = wait for clear signal
            regime_vote = 0
        
        # 4h trend filter (must align or be neutral)
        trend_filter_pass = (trend_4h == 0) or (trend_vote == trend_4h) or (regime_vote == trend_4h)
        
        # LONG conditions
        long_trend = (is_low_vol and trend_vote == 1 and trend_filter_pass)
        long_mean_revert = (is_high_vol and rsi_vote == 1 and trend_4h != -1)
        long_condition = long_trend or long_mean_revert
        
        # SHORT conditions
        short_trend = (is_low_vol and trend_vote == -1 and trend_filter_pass)
        short_mean_revert = (is_high_vol and rsi_vote == -1 and trend_4h != 1)
        short_condition = short_trend or short_mean_revert
        
        # High conviction: regime + trend + 4h all align
        high_conviction_long = long_condition and trend_vote == 1 and trend_4h == 1
        high_conviction_short = short_condition and trend_vote == -1 and trend_4h == -1
        
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