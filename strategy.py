#!/usr/bin/env python3
"""
EXPERIMENT #051 - REGIME ADAPTIVE ENSEMBLE (1h+4h with BBW Volatility Filter)
==================================================================================================
Hypothesis: Multi-timeframe strategies work well, but adding regime detection can improve
risk-adjusted returns by adapting to market conditions. This strategy combines:

Key innovations from #050:
- REGIME DETECTION: Bollinger Band Width percentile to identify low/high volatility
- ENSEMBLE VOTING: 3 signal types (HMA trend, Supertrend, RSI momentum) vote on direction
- ADAPTIVE SIZING: Position size scales with signal confidence (more agreement = larger size)
- VOLATILITY ADJUSTMENT: Reduce size in high volatility regimes to control drawdown

Why this should beat #050 (Sharpe=9.612):
- Regime filtering avoids bad trades in choppy/high-vol conditions
- Ensemble voting reduces false signals (need 2/3 agreement minimum)
- Adaptive sizing maximizes returns when confidence is high
- Based on proven 1h+4h MTF framework from #050

Position sizing rules (CRITICAL):
- MAX signal: 0.40 (never exceed)
- Base size: 0.25 (conservative)
- Confidence bonus: +0.05 per additional agreeing signal
- Volatility penalty: -0.10 in high vol regime
- Final range: 0.15 to 0.40
"""

import numpy as np
import pandas as pd

name = "regime_ensemble_hma_st_rsi_bbw_1h_4h_v1"
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
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = sma[i] + std_mult * std
        lower[i] = sma[i] - std_mult * std
        if sma[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / sma[i]
    
    return upper, lower, bbw


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
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        mean = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        if std > 0:
            zscore[i] = (close[i] - mean) / std
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    supertrend_1h, st_direction_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_1h = calculate_adx(high, low, close, period=14)
    _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    zscore_1h = calculate_zscore(close, period=20)
    
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
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    
    # Map 4h indicators back to 1h timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            st_trend_4h[i] = st_direction_4h[idx_4h]
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
    
    # Calculate BBW percentile for regime detection (rolling 100-period)
    bbw_percentile = np.zeros(n)
    bbw_window = 100
    for i in range(bbw_window - 1, n):
        valid_bbw = bbw_4h_mapped[i - bbw_window + 1:i + 1]
        valid_bbw = valid_bbw[valid_bbw > 0]
        if len(valid_bbw) > 0:
            current_bbw = bbw_4h_mapped[i]
            bbw_percentile[i] = np.sum(valid_bbw <= current_bbw) / len(valid_bbw)
    
    # Generate signals with ensemble voting and regime adaptation
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels with adaptive confidence
    SIZE_BASE = 0.25
    SIZE_CONFIDENCE_BONUS = 0.075  # Per additional agreeing signal
    SIZE_HIGH_VOL_PENALTY = 0.10
    SIZE_MAX = 0.40
    SIZE_MIN = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # BBW percentile threshold for regime (low vol = trend, high vol = mean revert)
    BBW_LOW_VOL_THRESHOLD = 0.40  # Below 40th percentile = low vol (trend regime)
    BBW_HIGH_VOL_THRESHOLD = 0.70  # Above 70th percentile = high vol (mean revert regime)
    
    first_valid = max(200, 40 * bars_per_4h, bbw_window)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_pct = bbw_percentile[i]
        zscore_val = zscore_1h[i]
        
        # Determine regime
        is_low_vol = bbw_pct < BBW_LOW_VOL_THRESHOLD
        is_high_vol = bbw_pct > BBW_HIGH_VOL_THRESHOLD
        is_mid_vol = not is_low_vol and not is_high_vol
        
        # Ensemble voting: 3 signal types
        # 1. HMA trend signal
        hma_signal = 0
        if trend == 1:
            hma_signal = 1
        elif trend == -1:
            hma_signal = -1
        
        # 2. Supertrend signal
        st_signal = 0
        if st_trend == 1:
            st_signal = 1
        elif st_trend == -1:
            st_signal = -1
        
        # 3. RSI momentum signal (with regime adaptation)
        rsi_signal = 0
        if is_low_vol or is_mid_vol:
            # Trend regime: RSI pullback
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and trend == 1:
                rsi_signal = 1
            elif RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and trend == -1:
                rsi_signal = -1
        else:
            # High vol regime: Mean reversion
            if zscore_val < -1.5:
                rsi_signal = 1
            elif zscore_val > 1.5:
                rsi_signal = -1
        
        # Count agreeing signals
        if hma_signal == 1 and st_signal == 1 and rsi_signal == 1:
            vote_count = 3
            vote_direction = 1
        elif hma_signal == -1 and st_signal == -1 and rsi_signal == -1:
            vote_count = 3
            vote_direction = -1
        elif (hma_signal == 1 and st_signal == 1) or (hma_signal == 1 and rsi_signal == 1) or (st_signal == 1 and rsi_signal == 1):
            vote_count = 2
            vote_direction = 1
        elif (hma_signal == -1 and st_signal == -1) or (hma_signal == -1 and rsi_signal == -1) or (st_signal == -1 and rsi_signal == -1):
            vote_count = 2
            vote_direction = -1
        else:
            vote_count = 0
            vote_direction = 0
        
        # ADX filter (4h) - only trade when trend is strong enough (in low/mid vol)
        if adx_4h_val < ADX_MIN and is_low_vol:
            vote_count = 0
        
        # Calculate position size based on confidence and regime
        if vote_count >= 2:
            base_size = SIZE_BASE
            confidence_bonus = (vote_count - 2) * SIZE_CONFIDENCE_BONUS
            vol_penalty = SIZE_HIGH_VOL_PENALTY if is_high_vol else 0.0
            
            position_size = base_size + confidence_bonus - vol_penalty
            position_size = np.clip(position_size, SIZE_MIN, SIZE_MAX)
            position_size = vote_direction * position_size
        else:
            position_size = 0.0
        
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
            
            # Stoploss check (2.5*ATR)
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
                    signals[i] = prev_side * 0.5 * abs(signals[i - 1]) if signals[i - 1] != 0 else 0.0
                    position_side[i] = prev_side
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
                    signals[i] = prev_side * 0.5 * abs(signals[i - 1]) if signals[i - 1] != 0 else 0.0
                    position_side[i] = prev_side
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
            
            # Hold position if no exit triggered and new signal agrees
            if vote_direction == prev_side or vote_count == 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Signal reversal - close position
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = False
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # Entry logic: Ensemble voting with minimum 2/3 agreement
        if vote_count >= 2:
            signals[i] = position_size
            position_side[i] = vote_direction
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals