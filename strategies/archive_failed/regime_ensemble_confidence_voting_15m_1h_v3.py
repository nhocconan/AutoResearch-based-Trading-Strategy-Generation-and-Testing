#!/usr/bin/env python3
"""
EXPERIMENT #094 - REGIME ADAPTIVE ENSEMBLE WITH CONFIDENCE VOTING (15m+1h v3)
==================================================================================================
Hypothesis: Experiment #090 proved regime+ensemble+voting on 1h+4h works (Sharpe=8.385).
Current best (#040) uses 15m+1h with Sharpe=16.016. This combines both approaches.

Key innovations:
1. THREE independent signal generators (trend, momentum, mean-reversion) with voting
2. Regime detection via BBW percentile: low vol=trend follow, high vol=mean revert
3. Confidence-based sizing: more signals agree = larger position (0.20 to 0.35)
4. 15m entries + 1h trend filter (proven in #031, #034, #035, #040)
5. Hysteresis on signal changes to reduce churn costs
6. Dynamic stoploss based on ATR regime (wider stops in high vol)

Why this should beat #040:
- Ensemble voting reduces false signals from any single indicator
- Regime adaptation switches strategy based on market conditions
- Confidence sizing maximizes returns when signals agree, minimizes risk when uncertain
- Based on #090's success with regime+ensemble but cleaner implementation
"""

import numpy as np
import pandas as pd

name = "regime_ensemble_confidence_voting_15m_1h_v3"
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    # EMA fast
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    # EMA slow
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    # MACD line
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal line
    valid_macd = macd_line[slow - 1:]
    if len(valid_macd) >= signal:
        signal_line[slow - 1 + signal - 1] = np.mean(valid_macd[:signal])
        for i in range(signal, len(valid_macd)):
            idx = slow - 1 + i
            signal_line[idx] = signal_line[idx - 1] + (2.0 / (signal + 1)) * (macd_line[idx] - signal_line[idx - 1])
    
    # Histogram
    for i in range(slow - 1, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


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
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # Resample to 1h for trend filters (4 x 15m = 1h)
    bars_per_1h = 4
    n_1h = (n // bars_per_1h)
    
    # Create 1h arrays by downsampling
    c_1h = np.zeros(n_1h)
    h_1h = np.zeros(n_1h)
    l_1h = np.zeros(n_1h)
    
    for i in range(n_1h):
        start_idx = i * bars_per_1h
        end_idx = start_idx + bars_per_1h
        c_1h[i] = close[end_idx - 1]
        h_1h[i] = np.max(high[start_idx:end_idx])
        l_1h[i] = np.min(low[start_idx:end_idx])
    
    # 1h indicators for trend
    hma_1h = calculate_hma(c_1h, period=21)
    supertrend_1h, st_direction_1h = calculate_supertrend(h_1h, l_1h, c_1h, period=10, multiplier=3.0)
    _, _, _, bbw_1h = calculate_bollinger_bands(c_1h, period=20, std_mult=2.0)
    bbw_pct_1h = calculate_bbw_percentile(bbw_1h, lookback=100)
    
    # Map 1h indicators back to 15m timeframe
    trend_1h = np.zeros(n)
    st_trend_1h = np.zeros(n)
    bbw_pct_1h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_1h = i // bars_per_1h
        if idx_1h < n_1h and idx_1h >= 40:
            if c_1h[idx_1h] > hma_1h[idx_1h]:
                trend_1h[i] = 1
            elif c_1h[idx_1h] < hma_1h[idx_1h]:
                trend_1h[i] = -1
            
            st_trend_1h[i] = st_direction_1h[idx_1h]
            bbw_pct_1h_mapped[i] = bbw_pct_1h[idx_1h]
    
    # Generate signals with ensemble voting
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on confidence
    SIZE_LOW = 0.20    # 1 signal agrees
    SIZE_MED = 0.28    # 2 signals agree
    SIZE_HIGH = 0.35   # 3 signals agree (max)
    
    # Regime thresholds (BBW percentile)
    REGIME_LOW_VOL = 0.30   # Below this = trend following regime
    REGIME_HIGH_VOL = 0.70  # Above this = mean reversion regime
    
    # ATR stoploss multiplier (dynamic based on regime)
    ATR_STOP_MULT_LOW = 2.0   # Tighter stops in low vol
    ATR_STOP_MULT_HIGH = 3.0  # Wider stops in high vol
    
    first_valid = max(200, 40 * bars_per_1h, 14 * 2, 20, 100)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Signal hysteresis to reduce churn
    prev_signal_vote = 0
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        st_trend = st_trend_1h[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        bbw_pct = bbw_pct_1h_mapped[i]
        macd_hist = macd_hist_15m[i]
        
        # Determine regime
        if bbw_pct < REGIME_LOW_VOL:
            regime = "trend"
            atr_mult = ATR_STOP_MULT_LOW
        elif bbw_pct > REGIME_HIGH_VOL:
            regime = "mean_revert"
            atr_mult = ATR_STOP_MULT_HIGH
        else:
            regime = "neutral"
            atr_mult = ATR_STOP_MULT_LOW
        
        # Check stoploss and take profit for existing positions
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
                stoploss_price = prev_entry - atr_mult * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    prev_signal_vote = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * atr_mult * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = signals[i - 1] * 0.5
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - atr_mult * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        prev_signal_vote = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + atr_mult * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    prev_signal_vote = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * atr_mult * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = signals[i - 1] * 0.5
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + atr_mult * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        prev_signal_vote = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ENSEMBLE VOTING: 3 independent signals
        vote_count = 0
        signal_direction = 0
        
        # Signal 1: TREND (HMA + Supertrend on 1h)
        if regime == "trend" or regime == "neutral":
            if trend == 1 and st_trend == 1:
                vote_count += 1
                signal_direction += 1
            elif trend == -1 and st_trend == -1:
                vote_count += 1
                signal_direction -= 1
        
        # Signal 2: MOMENTUM (RSI + MACD on 15m)
        if regime == "trend" or regime == "neutral":
            if rsi_val > 50 and macd_hist > 0:
                vote_count += 1
                signal_direction += 1
            elif rsi_val < 50 and macd_hist < 0:
                vote_count += 1
                signal_direction -= 1
        elif regime == "mean_revert":
            if rsi_val < 35:
                vote_count += 1
                signal_direction += 1
            elif rsi_val > 65:
                vote_count += 1
                signal_direction -= 1
        
        # Signal 3: MEAN REVERSION (Z-score on 15m)
        if regime == "mean_revert":
            if zscore_val < -1.5:
                vote_count += 1
                signal_direction += 1
            elif zscore_val > 1.5:
                vote_count += 1
                signal_direction -= 1
        elif regime == "trend" or regime == "neutral":
            # In trend regime, only take MR signals that align with trend
            if trend == 1 and zscore_val < -0.5:
                vote_count += 1
                signal_direction += 1
            elif trend == -1 and zscore_val > 0.5:
                vote_count += 1
                signal_direction -= 1
        
        # Determine signal based on vote count and direction
        if vote_count >= 2 and signal_direction != 0:
            # Hysteresis: require vote count to increase for new position
            if signal_direction > 0:
                if prev_signal_vote >= 0 or vote_count >= 3:
                    if vote_count == 3:
                        signals[i] = SIZE_HIGH
                    elif vote_count == 2:
                        signals[i] = SIZE_MED
                    else:
                        signals[i] = SIZE_LOW
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    prev_signal_vote = signal_direction * vote_count
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            elif signal_direction < 0:
                if prev_signal_vote <= 0 or vote_count >= 3:
                    if vote_count == 3:
                        signals[i] = -SIZE_HIGH
                    elif vote_count == 2:
                        signals[i] = -SIZE_MED
                    else:
                        signals[i] = -SIZE_LOW
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    prev_signal_vote = signal_direction * vote_count
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
            prev_signal_vote = 0
    
    return signals