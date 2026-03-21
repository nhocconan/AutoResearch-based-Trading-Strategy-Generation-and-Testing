#!/usr/bin/env python3
"""
EXPERIMENT #076 - ENSEMBLE_REGIME_CONFIDENCE_3SIGNAL_15M_4H_V1
==================================================================================================
Hypothesis: Combine 3 independent signal types (trend, momentum, mean-reversion) with regime-adaptive
weighting. Each signal type votes independently, and we weight by confidence (agreement level).
Regime detection via BBW percentile determines which signal type gets higher weight.

Key innovations from #070 (Sharpe=1.256):
- 3-signal ensemble instead of 2 (adds mean-reversion for choppy markets)
- Regime-adaptive weighting (trend signals weighted higher in low vol, MR in high vol)
- Confidence scoring based on signal agreement (more agreement = larger position)
- 15m entries with 4h trend filter (proven MTF combination)
- Conservative sizing: max 0.35, discrete levels (0.0, ±0.20, ±0.35)

Why this should beat current best:
- More robust across different market regimes (trending vs choppy)
- Reduces false signals by requiring multiple confirmations
- Adaptive weighting prevents overtrading in wrong regime
- Based on #070's successful KAMA+Donchian+regime approach
"""

import numpy as np
import pandas as pd

name = "ensemble_regime_confidence_3signal_15m_4h_v1"
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
    
    # Calculate EMAs
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    # MACD line
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal line
    valid_macd_start = slow - 1 + signal - 1
    if valid_macd_start < n:
        signal_line[valid_macd_start] = np.mean(macd_line[slow - 1:valid_macd_start + 1])
        for i in range(valid_macd_start + 1, n):
            signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    # Histogram
    for i in range(valid_macd_start, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower


def resample_to_timeframe(close, high, low, bars_per_tf):
    """Resample data to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return np.zeros(1), np.zeros(1), np.zeros(1)
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        c_tf[i] = close[end_idx - 1]
        h_tf[i] = np.max(high[start_idx:end_idx])
        l_tf[i] = np.min(low[start_idx:end_idx])
    
    return c_tf, h_tf, l_tf


def map_tf_to_base(tf_array, bars_per_tf, base_length):
    """Map higher timeframe array back to base timeframe"""
    mapped = np.zeros(base_length)
    n_tf = len(tf_array)
    
    for i in range(base_length):
        tf_idx = i // bars_per_tf
        if tf_idx < n_tf:
            mapped[i] = tf_array[tf_idx]
    
    return mapped


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
    
    # Base timeframe (15m) indicators
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    bb_upper_15m, bb_mid_15m, bb_lower_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    macd_15m, macd_sig_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    donch_upper_15m, donch_mid_15m, donch_lower_15m = calculate_donchian(high, low, period=20)
    
    # 4h timeframe (16 x 15m = 4h)
    bars_per_4h = 16
    c_4h, h_4h, l_4h = resample_to_timeframe(close, high, low, bars_per_4h)
    
    # 4h indicators for trend filter
    hma_4h = calculate_hma(c_4h, period=21)
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Map 4h indicators to 15m
    hma_4h_mapped = map_tf_to_base(hma_4h, bars_per_4h, n)
    kama_4h_mapped = map_tf_to_base(kama_4h, bars_per_4h, n)
    st_direction_4h_mapped = map_tf_to_base(st_direction_4h, bars_per_4h, n)
    adx_4h_mapped = map_tf_to_base(adx_4h, bars_per_4h, n)
    bbw_4h_mapped = map_tf_to_base(bbw_4h, bars_per_4h, n)
    bbw_pct_4h_mapped = map_tf_to_base(bbw_pct_4h, bars_per_4h, n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.10
    
    # Signal thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    ZSCORE_MAX = 2.0
    ADX_MIN = 20
    BBW_REGIME_LOW = 0.30  # Below 30th percentile = low vol (trend regime)
    BBW_REGIME_HIGH = 0.70  # Above 70th percentile = high vol (mean reversion regime)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Minimum warmup period
    first_valid = max(200, 100 * bars_per_4h, 14 * 2, 20, 28, 45)
    
    # Track position state
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for invalid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get current values
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_pct = bbw_pct_4h_mapped[i]
        
        # 4h trend signals
        hma_trend_4h = 1 if c_4h[i // bars_per_4h] > hma_4h[i // bars_per_4h] else (-1 if c_4h[i // bars_per_4h] < hma_4h[i // bars_per_4h] else 0)
        kama_trend_4h = 1 if c_4h[i // bars_per_4h] > kama_4h[i // bars_per_4h] else (-1 if c_4h[i // bars_per_4h] < kama_4h[i // bars_per_4h] else 0)
        st_trend_4h = st_direction_4h_mapped[i]
        
        # 15m signals
        hma_trend_15m = 1 if close[i] > hma_15m[i] else (-1 if close[i] < hma_15m[i] else 0)
        st_trend_15m = st_direction_15m[i]
        macd_signal_15m = 1 if macd_hist_15m[i] > 0 else (-1 if macd_hist_15m[i] < 0 else 0)
        rsi_signal_15m = 1 if rsi_val > 50 else (-1 if rsi_val < 50 else 0)
        
        # Regime detection
        is_trend_regime = bbw_pct < BBW_REGIME_LOW
        is_mr_regime = bbw_pct > BBW_REGIME_HIGH
        is_neutral_regime = not is_trend_regime and not is_mr_regime
        
        # === SIGNAL 1: TREND FOLLOWING (HMA + Supertrend + KAMA) ===
        trend_signal = 0
        trend_confidence = 0
        
        if is_trend_regime or is_neutral_regime:
            # Count trend agreements on 4h
            trend_votes = 0
            if hma_trend_4h == 1:
                trend_votes += 1
            if kama_trend_4h == 1:
                trend_votes += 1
            if st_trend_4h == 1:
                trend_votes += 1
            
            if trend_votes >= 2:
                trend_signal = 1
                trend_confidence = trend_votes / 3.0
            elif trend_votes <= 1:
                trend_votes_neg = 0
                if hma_trend_4h == -1:
                    trend_votes_neg += 1
                if kama_trend_4h == -1:
                    trend_votes_neg += 1
                if st_trend_4h == -1:
                    trend_votes_neg += 1
                if trend_votes_neg >= 2:
                    trend_signal = -1
                    trend_confidence = trend_votes_neg / 3.0
        
        # === SIGNAL 2: MOMENTUM (MACD + RSI + ADX) ===
        momentum_signal = 0
        momentum_confidence = 0
        
        if adx_4h_val > ADX_MIN:
            mom_votes = 0
            if macd_signal_15m == 1:
                mom_votes += 1
            if rsi_signal_15m == 1:
                mom_votes += 1
            if hma_trend_15m == 1:
                mom_votes += 1
            
            if mom_votes >= 2:
                momentum_signal = 1
                momentum_confidence = mom_votes / 3.0
            elif mom_votes <= 1:
                mom_votes_neg = 0
                if macd_signal_15m == -1:
                    mom_votes_neg += 1
                if rsi_signal_15m == -1:
                    mom_votes_neg += 1
                if hma_trend_15m == -1:
                    mom_votes_neg += 1
                if mom_votes_neg >= 2:
                    momentum_signal = -1
                    momentum_confidence = mom_votes_neg / 3.0
        
        # === SIGNAL 3: MEAN REVERSION (Z-score + Bollinger + RSI extremes) ===
        mr_signal = 0
        mr_confidence = 0
        
        if is_mr_regime or is_neutral_regime:
            mr_votes = 0
            if zscore_val < -1.5 and rsi_val < 40:
                mr_votes += 1
            if price < bb_lower_15m[i]:
                mr_votes += 1
            if rsi_val < 35:
                mr_votes += 1
            
            if mr_votes >= 2:
                mr_signal = 1
                mr_confidence = mr_votes / 3.0
            elif mr_votes <= 1:
                mr_votes_neg = 0
                if zscore_val > 1.5 and rsi_val > 60:
                    mr_votes_neg += 1
                if price > bb_upper_15m[i]:
                    mr_votes_neg += 1
                if rsi_val > 65:
                    mr_votes_neg += 1
                if mr_votes_neg >= 2:
                    mr_signal = -1
                    mr_confidence = mr_votes_neg / 3.0
        
        # === ENSEMBLE VOTING WITH REGIME WEIGHTING ===
        # Weight signals based on regime
        if is_trend_regime:
            trend_weight = 0.6
            momentum_weight = 0.3
            mr_weight = 0.1
        elif is_mr_regime:
            trend_weight = 0.2
            momentum_weight = 0.3
            mr_weight = 0.5
        else:
            trend_weight = 0.4
            momentum_weight = 0.4
            mr_weight = 0.2
        
        # Calculate weighted ensemble signal
        ensemble_score = (
            trend_signal * trend_confidence * trend_weight +
            momentum_signal * momentum_confidence * momentum_weight +
            mr_signal * mr_confidence * mr_weight
        )
        
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
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF * np.sign(prev_side)
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered and ensemble agrees
            if (prev_side == 1 and ensemble_score > 0.1) or (prev_side == -1 and ensemble_score < -0.1):
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Exit if ensemble disagrees
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # Entry logic based on ensemble score
        if ensemble_score > 0.25:
            # Scale position by confidence
            if abs(ensemble_score) > 0.4:
                signals[i] = SIZE_FULL
            elif abs(ensemble_score) > 0.25:
                signals[i] = SIZE_HALF
            else:
                signals[i] = SIZE_QUARTER
            
            position_side[i] = 1 if ensemble_score > 0 else -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif ensemble_score < -0.25:
            # Scale position by confidence
            if abs(ensemble_score) > 0.4:
                signals[i] = -SIZE_FULL
            elif abs(ensemble_score) > 0.25:
                signals[i] = -SIZE_HALF
            else:
                signals[i] = -SIZE_QUARTER
            
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals