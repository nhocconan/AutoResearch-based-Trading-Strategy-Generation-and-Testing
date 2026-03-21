#!/usr/bin/env python3
"""
EXPERIMENT #067 - SIMPLIFIED_REGIME_ENSEMBLE_15M_4H_V1
==================================================================================================
Hypothesis: Recent experiments #064-#066 show declining Sharpe (0.55→0.36→0.07) due to over-complexity.
This strategy simplifies the ensemble while keeping proven elements:
- 3 core signals only (Trend, Momentum, Volatility) instead of 5+
- Cleaner regime detection using BBW percentile + ADX
- Discrete position sizing (0, 0.20, 0.35) to reduce churn costs
- Proper 15m entries with 4h trend filter (proven in #055, #060, #062)
- Simplified hysteresis to prevent rapid flipping
- Trailing stoploss at 2*ATR, take profit at 3R

Key changes from #066:
- Removed RSI divergence (noisy, caused whipsaws)
- Removed confidence weighting (overfitting)
- Simplified to 3 signals with clear rules
- Stronger ADX filter (min 25 instead of 20)
- Better stoploss logic (trailing from entry, not dynamic)

Timeframe: 15m entries + 4h trend filter
Position Size: 0.0, 0.20, 0.35 (discrete levels)
Stoploss: 2*ATR trailing
Take Profit: 3R → close position
"""

import numpy as np
import pandas as pd

name = "simplified_regime_ensemble_15m_4h_v1"
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


def calculate_bbw_percentile(bbw, period=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(period - 1, n):
        window = bbw[i - period + 1:i + 1]
        current = bbw[i]
        percentile[i] = np.sum(window <= current) / period
    
    return percentile


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    er = np.zeros(n)
    
    for i in range(period - 1, n):
        change = abs(close[i] - close[i - period + 1])
        volatility = np.sum(np.abs(np.diff(close[i - period + 1:i + 1])))
        
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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
    upper_15m, middle_15m, lower_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, period=100)
    kama_15m = calculate_kama(close, period=10)
    
    # Resample to 4h for trend filters (16 x 15m = 4h)
    bars_per_4h = 16
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
    kama_4h = calculate_kama(c_4h, period=10)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            st_trend_4h[i] = st_direction_4h[idx_4h]
            adx_4h_mapped[i] = adx_4h[idx_4h]
    
    # Generate signals with simplified ensemble
    signals = np.zeros(n)
    
    # Position sizing - discrete levels
    SIZE_NONE = 0.0
    SIZE_SMALL = 0.20
    SIZE_LARGE = 0.35
    
    # Thresholds
    ADX_MIN = 25  # Stronger filter than before
    BBW_PCT_LOW = 0.30
    BBW_PCT_HIGH = 0.70
    ZSCORE_EXTREME = 2.0
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # Hysteresis - only flip on strong signal change
    HYSTERESIS = 0.25
    
    first_valid = max(200, 40 * bars_per_4h, 100)
    
    # Track position state for stoploss/TP
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Regime detection
        bbw_pct = bbw_pct_15m[i]
        adx_val = adx_4h_mapped[i]
        
        # Skip if ADX too low (no clear trend)
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Determine volatility regime
        if bbw_pct < BBW_PCT_LOW:
            regime = "low_vol"
        elif bbw_pct > BBW_PCT_HIGH:
            regime = "high_vol"
        else:
            regime = "med_vol"
        
        # Signal 1: Trend signal (4h HMA + Supertrend agreement)
        trend_signal = 0
        if trend_4h[i] == 1 and st_trend_4h[i] == 1:
            trend_signal = 1
        elif trend_4h[i] == -1 and st_trend_4h[i] == -1:
            trend_signal = -1
        
        # Signal 2: Momentum signal (15m HMA + KAMA + RSI)
        momentum_signal = 0
        rsi_val = rsi_15m[i]
        
        if close[i] > hma_15m[i] and close[i] > kama_15m[i] and rsi_val > 50:
            momentum_signal = 1
        elif close[i] < hma_15m[i] and close[i] < kama_15m[i] and rsi_val < 50:
            momentum_signal = -1
        
        # Signal 3: Volatility signal (15m Z-score + BB position)
        vol_signal = 0
        zscore_val = zscore_15m[i]
        bb_position = (close[i] - lower_15m[i]) / (upper_15m[i] - lower_15m[i] + 1e-10)
        
        if regime == "high_vol":
            # High vol = mean reversion
            if zscore_val < -ZSCORE_EXTREME and bb_position < 0.2:
                vol_signal = 1
            elif zscore_val > ZSCORE_EXTREME and bb_position > 0.8:
                vol_signal = -1
        else:
            # Low/med vol = trend following
            if zscore_val > 0.5 and bb_position > 0.6:
                vol_signal = 1
            elif zscore_val < -0.5 and bb_position < 0.4:
                vol_signal = -1
        
        # Ensemble voting - need 2+ signals agreeing
        vote_sum = trend_signal + momentum_signal + vol_signal
        agreement_count = sum([
            1 if trend_signal != 0 else 0,
            1 if momentum_signal != 0 else 0,
            1 if vol_signal != 0 else 0
        ])
        
        # Determine target signal
        target_signal = SIZE_NONE
        if agreement_count >= 2:
            if vote_sum >= 2:
                target_signal = SIZE_LARGE
            elif vote_sum <= -2:
                target_signal = -SIZE_LARGE
            elif vote_sum == 1:
                target_signal = SIZE_SMALL
            elif vote_sum == -1:
                target_signal = -SIZE_SMALL
        
        # Apply hysteresis
        prev_signal = signals[i - 1] if i > 0 else 0.0
        if prev_signal != 0 and target_signal != 0:
            if np.sign(prev_signal) == np.sign(target_signal):
                # Same direction - keep position
                target_signal = prev_signal
            elif abs(target_signal) < abs(prev_signal) + HYSTERESIS:
                # Not enough change to flip
                target_signal = prev_signal
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            
            price = close[i]
            atr = atr_15m[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry, price)
                current_low = min(lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else price, price)
            else:
                current_high = max(highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else price, price)
                current_low = min(lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2*ATR trailing)
            stoploss_distance = 2.0 * atr
            
            if prev_side == 1:
                stoploss_price = current_high - stoploss_distance
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (3R)
                tp_price = prev_entry + 3 * stoploss_distance
                if price >= tp_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                    
            elif prev_side == -1:
                stoploss_price = current_low + stoploss_distance
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (3R)
                tp_price = prev_entry - 3 * stoploss_distance
                if price <= tp_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Apply target signal if no existing position
        signals[i] = target_signal
        if target_signal != 0:
            position_side[i] = np.sign(target_signal)
            entry_price[i] = close[i]
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        else:
            position_side[i] = 0
    
    return signals