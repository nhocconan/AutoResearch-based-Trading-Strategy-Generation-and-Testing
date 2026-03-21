#!/usr/bin/env python3
"""
EXPERIMENT #091 - Confidence-Weighted Ensemble with ADX Trend Strength (1h + 4h)
==================================================================================================
Hypothesis: Enhance ensemble voting by adding MACD as 4th signal, use ADX to filter weak trends,
and scale position size by signal confidence (more agreement = larger position). Add Z-score
filter to avoid entering at extreme extensions.

Key improvements over #090:
- 4-signal ensemble (Supertrend, HMA, KAMA, MACD) instead of 3
- ADX(14) > 25 filter to ensure strong trend before entering
- Z-score(20) filter: don't enter if |z| > 2.0 (overextended)
- Confidence-based sizing: 4/4 agree=0.35, 3/4=0.25, 2/4=0.15
- 1h entries with 4h trend filter + ADX confirmation
- Proper state tracking using lists (no read-only array issues)

Why this should work:
- MACD adds momentum confirmation missing from pure trend indicators
- ADX filters choppy markets where ensemble gives false signals
- Z-score prevents buying tops/selling bottoms
- Confidence sizing maximizes returns when signals strongly agree
- Based on successful #090 but with additional filters for quality
"""

import numpy as np
import pandas as pd

name = "confidence_ensemble_adx_zscore_1h_4h_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    # Calculate EMA fast
    multiplier_fast = 2.0 / (fast + 1)
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = (close[i] - ema_fast[i - 1]) * multiplier_fast + ema_fast[i - 1]
    
    # Calculate EMA slow
    multiplier_slow = 2.0 / (slow + 1)
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = (close[i] - ema_slow[i - 1]) * multiplier_slow + ema_slow[i - 1]
    
    # MACD line
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal line
    multiplier_signal = 2.0 / (signal_period + 1)
    first_signal = slow - 1 + signal_period - 1
    if first_signal < n:
        signal_line[first_signal] = np.mean(macd_line[slow - 1:first_signal + 1])
        for i in range(first_signal + 1, n):
            signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier_signal + signal_line[i - 1]
    
    # Histogram
    for i in range(n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
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
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initial sums
    sum_tr = np.sum(tr[1:period + 1])
    sum_plus_dm = np.sum(plus_dm[1:period + 1])
    sum_minus_dm = np.sum(minus_dm[1:period + 1])
    
    for i in range(period, n):
        if i == period:
            pass
        else:
            sum_tr = sum_tr - tr[i] + tr[i]
            sum_plus_dm = sum_plus_dm - plus_dm[i] + plus_dm[i]
            sum_minus_dm = sum_minus_dm - minus_dm[i] + minus_dm[i]
        
        if sum_tr > 0:
            plus_di[i] = 100 * sum_plus_dm / sum_tr
            minus_di[i] = 100 * sum_minus_dm / sum_tr
        else:
            plus_di[i] = 0
            minus_di[i] = 0
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # ADX = SMA of DX
    for i in range(period * 2 - 1, n):
        adx[i] = np.mean(dx[i - period + 1:i + 1])
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price)"""
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
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    hma_1h = calculate_hma(close, period=21)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    supertrend_1h, st_direction_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    macd_1h, signal_1h, hist_1h = calculate_macd(close, fast=12, slow=26, signal_period=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    bbw_pct_1h = calculate_bbw_percentile(bbw_1h, lookback=100)
    
    # Resample to 4h for trend filters (4 x 1h = 4h)
    bars_per_4h = 4
    n_4h = (n // bars_per_4h)
    
    # Create 4h arrays by downsampling
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * bars_per_4h
        end_idx = start_idx + bars_per_4h
        c_4h[i] = close[end_idx - 1]
        h_4h[i] = np.max(high[start_idx:end_idx])
        l_4h[i] = np.min(low[start_idx:end_idx])
    
    # 4h indicators for trend
    hma_4h = calculate_hma(c_4h, period=21)
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    macd_4h, signal_4h, hist_4h = calculate_macd(c_4h, fast=12, slow=26, signal_period=9)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Map 4h indicators back to 1h timeframe
    trend_4h_hma = np.zeros(n)
    trend_4h_kama = np.zeros(n)
    trend_4h_st = np.zeros(n)
    trend_4h_macd = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h_hma[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h_hma[i] = -1
            
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                trend_4h_kama[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                trend_4h_kama[i] = -1
            
            trend_4h_st[i] = st_direction_4h[idx_4h]
            
            if hist_4h[idx_4h] > 0:
                trend_4h_macd[i] = 1
            elif hist_4h[idx_4h] < 0:
                trend_4h_macd[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
    
    # 1h signal generators (4 independent signals for voting)
    signal_supertrend = np.zeros(n)
    signal_hma = np.zeros(n)
    signal_kama = np.zeros(n)
    signal_macd = np.zeros(n)
    
    for i in range(100, n):
        # Supertrend signal (1h)
        signal_supertrend[i] = st_direction_1h[i]
        
        # HMA signal (1h)
        if close[i] > hma_1h[i]:
            signal_hma[i] = 1
        elif close[i] < hma_1h[i]:
            signal_hma[i] = -1
        
        # KAMA signal (1h)
        if close[i] > kama_1h[i]:
            signal_kama[i] = 1
        elif close[i] < kama_1h[i]:
            signal_kama[i] = -1
        
        # MACD signal (1h)
        if hist_1h[i] > 0:
            signal_macd[i] = 1
        elif hist_1h[i] < 0:
            signal_macd[i] = -1
    
    # Generate signals with ensemble voting
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on confidence
    SIZE_4OF4 = 0.35  # All 4 signals agree
    SIZE_3OF4 = 0.25  # 3 of 4 signals agree
    SIZE_2OF4 = 0.15  # 2 of 4 signals agree (minimum)
    SIZE_HALF_4OF4 = 0.175
    SIZE_HALF_3OF4 = 0.125
    SIZE_HALF_2OF4 = 0.075
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # ADX threshold for strong trend
    ADX_MIN = 25.0
    
    # Z-score threshold (don't enter if overextended)
    ZSCORE_MAX = 2.0
    
    first_valid = max(200, 40 * bars_per_4h, 100)
    
    # Track position state using lists to avoid read-only issues
    position_side = [0] * n
    entry_price = [0.0] * n
    tp_triggered = [0] * n
    highest_since_entry = [0.0] * n
    lowest_since_entry = [0.0] * n
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Regime detection (4h BBW percentile)
        regime_low_vol = bbw_pct_4h[i] < 0.50
        
        # 4h trend filter - need at least 3/4 indicators to agree
        trend_votes_long = 0
        trend_votes_short = 0
        
        if trend_4h_hma[i] == 1:
            trend_votes_long += 1
        elif trend_4h_hma[i] == -1:
            trend_votes_short += 1
        
        if trend_4h_kama[i] == 1:
            trend_votes_long += 1
        elif trend_4h_kama[i] == -1:
            trend_votes_short += 1
        
        if trend_4h_st[i] == 1:
            trend_votes_long += 1
        elif trend_4h_st[i] == -1:
            trend_votes_short += 1
        
        if trend_4h_macd[i] == 1:
            trend_votes_long += 1
        elif trend_4h_macd[i] == -1:
            trend_votes_short += 1
        
        # Determine 4h trend direction
        trend_4h = 0
        if trend_votes_long >= 3:
            trend_4h = 1
        elif trend_votes_short >= 3:
            trend_4h = -1
        
        # ADX filter - only trade if trend is strong enough
        adx_strong = adx_4h_mapped[i] > ADX_MIN
        
        # Z-score filter - don't enter if overextended
        zscore_ok = abs(zscore_1h[i]) < ZSCORE_MAX
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, close[i])
                current_low = min(prev_low, close[i]) if prev_low > 0 else close[i]
            else:
                current_high = max(prev_high, close[i]) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and close[i] >= tp_price:
                    # Determine half size based on original confidence
                    if abs(signals[i - 1]) >= SIZE_4OF4 - 0.01:
                        signals[i] = SIZE_HALF_4OF4
                    elif abs(signals[i - 1]) >= SIZE_3OF4 - 0.01:
                        signals[i] = SIZE_HALF_3OF4
                    else:
                        signals[i] = SIZE_HALF_2OF4
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and close[i] <= tp_price:
                    # Determine half size based on original confidence
                    if abs(signals[i - 1]) >= SIZE_4OF4 - 0.01:
                        signals[i] = -SIZE_HALF_4OF4
                    elif abs(signals[i - 1]) >= SIZE_3OF4 - 0.01:
                        signals[i] = -SIZE_HALF_3OF4
                    else:
                        signals[i] = -SIZE_HALF_2OF4
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if close[i] > trail_stop:
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
        
        # 1h ensemble voting: count agreement with 4h trend
        vote_count = 0
        
        if trend_4h == 1:
            if signal_supertrend[i] == 1:
                vote_count += 1
            if signal_hma[i] == 1:
                vote_count += 1
            if signal_kama[i] == 1:
                vote_count += 1
            if signal_macd[i] == 1:
                vote_count += 1
        elif trend_4h == -1:
            if signal_supertrend[i] == -1:
                vote_count += 1
            if signal_hma[i] == -1:
                vote_count += 1
            if signal_kama[i] == -1:
                vote_count += 1
            if signal_macd[i] == -1:
                vote_count += 1
        
        # Entry logic: 4h trend + ADX strong + Z-score ok + minimum 2/4 voting
        if trend_4h == 1 and adx_strong and zscore_ok and vote_count >= 2:
            # Determine position size by confidence
            if vote_count == 4:
                signals[i] = SIZE_4OF4
            elif vote_count == 3:
                signals[i] = SIZE_3OF4
            else:
                signals[i] = SIZE_2OF4
            
            position_side[i] = 1
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
                
        elif trend_4h == -1 and adx_strong and zscore_ok and vote_count >= 2:
            # Determine position size by confidence
            if vote_count == 4:
                signals[i] = -SIZE_4OF4
            elif vote_count == 3:
                signals[i] = -SIZE_3OF4
            else:
                signals[i] = -SIZE_2OF4
            
            position_side[i] = -1
            entry_price[i] = close[i]
            tp_triggered[i] = 0
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return np.array(signals)