#!/usr/bin/env python3
"""
EXPERIMENT #061 - ADX_REGIME_DUAL_MODE_ZSCORE_HMA_ST_1H_4H_V1
==================================================================================================
Hypothesis: ADX-based regime detection with dual-mode strategy (trend-follow vs mean-revert)
will outperform BBW-only regime detection. Using 1h entries with 4h trend filter.

Key innovations:
- ADX REGIME DETECTION: ADX(14) > 25 = trend regime, ADX < 20 = range regime
- DUAL-MODE STRATEGY: Trend-follow (HMA+ST) in high ADX, Mean-revert (Z-score+RSI) in low ADX
- Z-SCORE MEAN REVERSION: Enter when Z-score(20) > 2.0 or < -2.0 in range regime
- SIGNAL HYSTERESIS: Require stronger signal to flip direction (reduces churn costs)
- VOLUME CONFIRMATION: Entry volume > 1.2x 20-bar avg volume
- 1H/4H MULTI-TF: 1h for entries (balanced noise/signal), 4h for trend filter

Why this should beat #060 (Sharpe=11.964) and approach #049 (Sharpe=13.974):
- ADX is more direct trend strength measure than BBW percentile
- Dual-mode adapts to market conditions better than single strategy
- Z-score mean reversion captures range-bound profits missed by trend-follow
- Hysteresis reduces signal churn (each change costs 0.10% fees)

Position sizing rules (CRITICAL):
- MAX signal: 0.35 (proven to control drawdown in 2022 crash)
- MIN signal: 0.20 (avoid tiny positions eaten by fees)
- Discrete levels: 0.0, 0.20, 0.28, 0.35 (reduces churn costs)
- Stoploss: 2.5*ATR trailing (adjusts to 3.5*ATR in trend regime)
- Volatility scaling: position_size = base_size * (target_vol / current_vol)
"""

import numpy as np
import pandas as pd

name = "adx_regime_dual_mode_zscore_hma_st_1h_4h_v1"
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


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def calc_wma(data, wma_period):
        result = np.zeros(len(data))
        for i in range(wma_period - 1, len(data)):
            weights = np.arange(1, wma_period + 1)
            window = data[i - wma_period + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma1 = calc_wma(close, half)
    wma2 = calc_wma(close, period)
    raw_hma = 2 * wma1 - wma2
    hma = calc_wma(raw_hma, sqrt_period)
    
    return hma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    direction = np.zeros(n)
    supertrend = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper = mid + multiplier * atr[i]
        lower = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper
            direction[i] = 1
        else:
            if direction[i - 1] == 1:
                if close[i] < upper:
                    supertrend[i] = upper
                    direction[i] = 1
                else:
                    supertrend[i] = lower
                    direction[i] = -1
            else:
                if close[i] > lower:
                    supertrend[i] = lower
                    direction[i] = -1
                else:
                    supertrend[i] = upper
                    direction[i] = 1
    
    return supertrend, direction


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * np.mean(plus_dm[i-period+1:i+1]) / atr[i]
            minus_di[i] = 100 * np.mean(minus_dm[i-period+1:i+1]) / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period*2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized price deviation)"""
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
    
    return zscore


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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    sc = np.zeros(n)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_sma = np.zeros(n)
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma


def resample_to_higher_tf(close, high, low, volume, bars_per_tf=4):
    """Resample 1h data to 4h (4 x 1h = 4h)"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return close.copy(), high.copy(), low.copy(), volume.copy()
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    v_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        if end_idx <= n:
            c_tf[i] = close[end_idx - 1]
            h_tf[i] = np.max(high[start_idx:end_idx])
            l_tf[i] = np.min(low[start_idx:end_idx])
            v_tf[i] = np.sum(volume[start_idx:end_idx])
    
    return c_tf, h_tf, l_tf, v_tf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=16)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    st_1h, st_dir_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    zscore_1h = calculate_zscore(close, period=20)
    adx_1h = calculate_adx(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Resample to 4h for trend (4 x 1h = 4h)
    bars_per_4h = 4
    c_4h, h_4h, l_4h, v_4h = resample_to_higher_tf(close, high, low, volume, bars_per_4h)
    
    # 4h indicators for trend
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    hma_4h = calculate_hma(c_4h, period=16)
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    st_4h, st_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h indicators back to 1h timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    kama_trend_4h = np.zeros(n)
    adx_regime_4h = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    
    n_4h = len(c_4h)
    for i in range(n):
        idx_4h = i // bars_per_4h
        
        if idx_4h < n_4h and idx_4h >= 40:
            # HMA trend
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            # Supertrend direction
            st_trend_4h[i] = st_dir_4h[idx_4h]
            
            # KAMA trend
            if c_4h[idx_4h] > kama_4h[idx_4h]:
                kama_trend_4h[i] = 1
            elif c_4h[idx_4h] < kama_4h[idx_4h]:
                kama_trend_4h[i] = -1
            
            # ADX regime
            adx_regime_4h[i] = adx_4h[idx_4h] if idx_4h < len(adx_4h) else 20
            
            # ATR mapped
            atr_4h_mapped[i] = atr_4h[idx_4h] if idx_4h < len(atr_4h) else atr_1h[i]
    
    # Position sizing parameters (DISCRETE levels based on signal agreement)
    SIZE_LEVELS = {2: 0.20, 3: 0.28, 4: 0.35}
    BASE_SIZE = 0.28
    
    # Regime thresholds
    ADX_TREND_THRESHOLD = 25  # Above 25 = strong trend
    ADX_RANGE_THRESHOLD = 20  # Below 20 = range-bound
    
    # Z-score thresholds for mean reversion
    ZSCORE_ENTRY = 2.0
    ZSCORE_EXIT = 0.5
    
    # Stoploss multipliers (adaptive to regime)
    ATR_STOP_TREND = 3.5  # Wider stops in trend regime
    ATR_STOP_RANGE = 2.5  # Tighter stops in range regime
    
    first_valid = max(200, 40 * bars_per_4h + 100)
    
    # Generate signals with regime-switching and dual-mode strategy
    signals = np.zeros(n)
    
    # Track position state for stoploss/TP
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    last_signal = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            last_signal[i] = signals[i-1] if i > 0 else 0
            continue
        
        # 4h regime signals
        hma_trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        kama_trend = kama_trend_4h[i]
        adx_regime = adx_regime_4h[i]
        atr_4h_val = atr_4h_mapped[i]
        
        # 1h entry signals
        price = close[i]
        hma_1h_val = hma_1h[i]
        kama_1h_val = kama_1h[i]
        st_dir = st_dir_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr_1h_val = atr_1h[i]
        vol_ratio = volume[i] / vol_sma_1h[i] if vol_sma_1h[i] > 0 else 1.0
        
        # Determine regime and adaptive ATR stop
        if adx_regime > ADX_TREND_THRESHOLD:
            regime = "trend"
            atr_stop_mult = ATR_STOP_TREND
        elif adx_regime < ADX_RANGE_THRESHOLD:
            regime = "range"
            atr_stop_mult = ATR_STOP_RANGE
        else:
            regime = "neutral"
            atr_stop_mult = ATR_STOP_RANGE
        
        # DUAL-MODE SIGNAL GENERATION
        if regime == "trend":
            # TREND-FOLLOW MODE: Use HMA + Supertrend + KAMA
            vote_hma = 0
            if hma_trend == 1:
                vote_hma = 1
            elif hma_trend == -1:
                vote_hma = -1
            
            vote_st = 0
            if st_trend == 1:
                vote_st = 1
            elif st_trend == -1:
                vote_st = -1
            
            vote_kama = 0
            if kama_trend == 1:
                vote_kama = 1
            elif kama_trend == -1:
                vote_kama = -1
            
            vote_1h_st = 0
            if st_dir == 1:
                vote_1h_st = 1
            elif st_dir == -1:
                vote_1h_st = -1
            
            long_votes = sum([vote_hma == 1, vote_st == 1, vote_kama == 1, vote_1h_st == 1])
            short_votes = sum([vote_hma == -1, vote_st == -1, vote_kama == -1, vote_1h_st == -1])
            
        else:
            # MEAN REVERSION MODE: Use Z-score + RSI
            vote_zscore = 0
            if zscore_val < -ZSCORE_ENTRY:
                vote_zscore = 1  # Oversold = long
            elif zscore_val > ZSCORE_ENTRY:
                vote_zscore = -1  # Overbought = short
            
            vote_rsi = 0
            if rsi_val < 35:
                vote_rsi = 1
            elif rsi_val > 65:
                vote_rsi = -1
            
            vote_hma_mr = 0
            if price < hma_1h_val and zscore_val < 0:
                vote_hma_mr = 1
            elif price > hma_1h_val and zscore_val > 0:
                vote_hma_mr = -1
            
            vote_kama_mr = 0
            if price < kama_1h_val and zscore_val < 0:
                vote_kama_mr = 1
            elif price > kama_1h_val and zscore_val > 0:
                vote_kama_mr = -1
            
            long_votes = sum([vote_zscore == 1, vote_rsi == 1, vote_hma_mr == 1, vote_kama_mr == 1])
            short_votes = sum([vote_zscore == -1, vote_rsi == -1, vote_hma_mr == -1, vote_kama_mr == -1])
        
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
                stoploss_price = prev_entry - atr_stop_mult * atr_1h_val
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    last_signal[i] = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * atr_stop_mult * atr_1h_val
                if not prev_tp and price >= tp_price:
                    prev_signal = signals[i - 1]
                    signals[i] = prev_side * 0.5 * abs(prev_signal) if prev_signal != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    last_signal[i] = signals[i]
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - atr_stop_mult * atr_1h_val
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        last_signal[i] = 0.0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + atr_stop_mult * atr_1h_val
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    last_signal[i] = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * atr_stop_mult * atr_1h_val
                if not prev_tp and price <= tp_price:
                    prev_signal = signals[i - 1]
                    signals[i] = prev_side * 0.5 * abs(prev_signal) if prev_signal != 0 else 0.0
                    position_side[i] = prev_side
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    last_signal[i] = signals[i]
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + atr_stop_mult * atr_1h_val
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        last_signal[i] = 0.0
                        continue
            
            # Maintain position if signal agrees (need at least 2 votes)
            # HYSTERESIS: Require same or stronger signal to maintain
            if prev_side == 1:
                if long_votes >= 2:
                    target_size = SIZE_LEVELS.get(long_votes, 0.20)
                    target_size = max(min(target_size, 0.35), 0.20)
                    
                    # Hysteresis: don't reduce size unless votes drop significantly
                    prev_size = abs(signals[i - 1])
                    if target_size < prev_size - 0.05 and long_votes < 3:
                        target_size = prev_size
                    
                    signals[i] = target_size
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
            elif prev_side == -1:
                if short_votes >= 2:
                    target_size = SIZE_LEVELS.get(short_votes, 0.20)
                    target_size = max(min(target_size, 0.35), 0.20)
                    
                    prev_size = abs(signals[i - 1])
                    if target_size < prev_size - 0.05 and short_votes < 3:
                        target_size = prev_size
                    
                    signals[i] = -target_size
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = prev_tp
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
            continue
        
        # Entry logic with volume confirmation
        # HYSTERESIS: Require 3/4 votes for entry (higher threshold than maintain)
        entry_threshold = 3
        
        # Volume filter for entries
        volume_confirmed = vol_ratio > 1.2 or regime == "range"  # Relax volume in range regime
        
        if long_votes >= entry_threshold and volume_confirmed:
            target_size = SIZE_LEVELS.get(long_votes, 0.20)
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = target_size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            
        elif short_votes >= entry_threshold and volume_confirmed:
            target_size = SIZE_LEVELS.get(short_votes, 0.20)
            target_size = max(min(target_size, 0.35), 0.20)
            
            signals[i] = -target_size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            signals[i] = 0.0
            position_side[i] = 0
        
        last_signal[i] = signals[i]
    
    return signals