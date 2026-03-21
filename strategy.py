#!/usr/bin/env python3
"""
EXPERIMENT #069 - SIMPLIFIED_MTF_ENSEMBLE_1H_4H_V1
==================================================================================================
Hypothesis: Simpler 3-signal ensemble with stricter agreement requirements reduces churn and fees.
- Trend Signal: 4h HMA direction (primary trend filter)
- Momentum Signal: 1h RSI + MACD histogram agreement
- Volatility Signal: 1h ATR expansion confirmation
- Entry: All 3 signals must agree (stricter than weighted voting)
- Exit: ATR trailing stop + time-based exit (reduce holding period)

Why this should beat #068 (Sharpe=0.162):
- Less signal churn = fewer fees (0.10% per change adds up fast)
- Stricter agreement = higher quality trades
- 1h timeframe instead of 15m = less noise, cleaner signals
- Based on #058 pattern (KAMA+Donchian+ADX+Zscore had Sharpe=5.35)
- Simpler logic = fewer bugs (avoid #059 index errors)

Key changes from #068:
- 1h timeframe instead of 15m (8x fewer bars, 8x fewer fee opportunities)
- All 3 signals must agree (not weighted voting)
- Time-based exit after 20 bars (reduce exposure)
- Volatility-adjusted position sizing (smaller size in high vol)
"""

import numpy as np
import pandas as pd

name = "simplified_mtf_ensemble_1h_4h_v1"
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
    """Calculate MACD"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = (close[i] * (2 / (fast + 1))) + (ema_fast[i - 1] * (1 - 2 / (fast + 1)))
    
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = (close[i] * (2 / (slow + 1))) + (ema_slow[i - 1] * (1 - 2 / (slow + 1)))
    
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    valid_macd = macd_line[slow - 1:]
    if len(valid_macd) >= signal:
        signal_line[slow - 1 + signal - 1] = np.mean(valid_macd[:signal])
        for i in range(signal, len(valid_macd)):
            idx = slow - 1 + i
            signal_line[idx] = (valid_macd[i] * (2 / (signal + 1))) + (signal_line[idx - 1] * (1 - 2 / (signal + 1)))
    
    for i in range(slow - 1, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    noise_ratio = np.zeros(n)
    sc = np.zeros(n)
    
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if noise > 0:
            noise_ratio[i] = change / noise
        else:
            noise_ratio[i] = 0
        
        sc[i] = (noise_ratio[i] * (2 / (fast_period + 1) - 2 / (slow_period + 1)) + 2 / (slow_period + 1)) ** 2
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


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


def resample_to_timeframe(close, high, low, open_price, bars_per_tf):
    """Resample data to higher timeframe"""
    n = len(close)
    n_tf = n // bars_per_tf
    
    if n_tf < 1:
        return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    
    c_tf = np.zeros(n_tf)
    h_tf = np.zeros(n_tf)
    l_tf = np.zeros(n_tf)
    o_tf = np.zeros(n_tf)
    
    for i in range(n_tf):
        start_idx = i * bars_per_tf
        end_idx = start_idx + bars_per_tf
        c_tf[i] = close[end_idx - 1]
        h_tf[i] = np.max(high[start_idx:end_idx])
        l_tf[i] = np.min(low[start_idx:end_idx])
        o_tf[i] = open_price[start_idx]
    
    return c_tf, h_tf, l_tf, o_tf


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_price = prices["open"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    macd_1h, _, hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_1h = calculate_adx(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    _, bb_mid_1h, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Resample to 4h for trend regime (4 x 1h = 4h)
    bars_per_4h = 4
    c_4h, h_4h, l_4h, o_4h = resample_to_timeframe(close, high, low, open_price, bars_per_4h)
    n_4h = len(c_4h)
    
    # 4h indicators for trend regime
    hma_4h = calculate_hma(c_4h, period=21)
    st_4h, st_dir_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    
    # Map 4h indicators back to 1h timeframe
    trend_4h = np.zeros(n)
    st_trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    kama_trend_4h = np.zeros(n)
    
    # Calculate ATR percentile for volatility adjustment
    atr_valid = atr_1h[100:]
    atr_sorted = np.sort(atr_valid)
    atr_percentile = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 20:
            # Trend direction (HMA)
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            # Supertrend direction
            st_trend_4h[i] = st_dir_4h[idx_4h]
            adx_4h_mapped[i] = adx_4h[idx_4h]
            
            # KAMA trend
            if idx_4h >= 1:
                if c_4h[idx_4h] > kama_4h[idx_4h]:
                    kama_trend_4h[i] = 1
                elif c_4h[idx_4h] < kama_4h[idx_4h]:
                    kama_trend_4h[i] = -1
        
        # ATR percentile
        if i >= 100 and atr_1h[i] > 0:
            atr_percentile[i] = np.searchsorted(atr_sorted, atr_1h[i]) / len(atr_sorted)
    
    # Generate signals with strict ensemble logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels with volatility adjustment
    SIZE_BASE = 0.30
    SIZE_LOW = 0.20
    SIZE_HIGH = 0.35
    
    # Signal thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    ADX_MIN = 18
    MACD_HIST_MIN = 0
    ZSCORE_MAX = 1.8
    ATR_STOP_MULT = 2.5
    EXIT_BARS = 20  # Time-based exit
    
    first_valid = max(200, 50 * bars_per_4h, 100)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    entry_bar = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(zscore_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get regime info
        trend = trend_4h[i]
        st_trend = st_trend_4h[i]
        kama_trend = kama_trend_4h[i]
        adx_val = adx_4h_mapped[i]
        atr_pct = atr_percentile[i]
        
        # ADX filter - only trade when trend has strength
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss, take profit, and time exit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_entry_bar = entry_bar[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            atr = atr_1h[i]
            price = close[i]
            bars_held = i - prev_entry_bar
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Time-based exit (reduce position after 20 bars)
            if bars_held >= EXIT_BARS and not prev_tp:
                signals[i] = prev_side * SIZE_LOW
                position_side[i] = prev_side
                entry_price[i] = prev_entry
                entry_bar[i] = prev_entry_bar
                tp_triggered[i] = 1
                continue
            
            # Stoploss check
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_bar[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = prev_side * SIZE_LOW
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    entry_bar[i] = prev_entry_bar
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        entry_bar[i] = 0
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
                    entry_bar[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = prev_side * SIZE_LOW
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    entry_bar[i] = prev_entry_bar
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        entry_bar[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            entry_bar[i] = entry_bar[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ENSEMBLE SIGNAL GENERATION - All 3 must agree
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        macd_hist = hist_1h[i]
        price = close[i]
        bb_pos = (price - bb_mid_1h[i]) / (bb_mid_1h[i] * 0.02 + 1e-10)
        
        # Signal 1: 4h Trend (HMA + Supertrend + KAMA agreement)
        trend_agree = (trend == st_trend == kama_trend) and trend != 0
        
        # Signal 2: 1h Momentum (RSI + MACD agreement)
        if prev_side == 0:
            momentum_long = (rsi_val > RSI_LONG_MIN and rsi_val < RSI_LONG_MAX and macd_hist > MACD_HIST_MIN)
            momentum_short = (rsi_val > RSI_SHORT_MIN and rsi_val < RSI_SHORT_MAX and macd_hist < -MACD_HIST_MIN)
        else:
            momentum_long = (rsi_val > 45 and macd_hist > 0)
            momentum_short = (rsi_val < 55 and macd_hist < 0)
        momentum_agree = momentum_long or momentum_short
        
        # Signal 3: Volatility (Z-score not extreme, ATR not too high)
        vol_ok = (abs(zscore_val) < ZSCORE_MAX) and (atr_pct < 0.85)
        
        # STRICT ENSEMBLE: All 3 signals must agree
        if trend_agree and momentum_agree and vol_ok:
            if trend == 1 and momentum_long:
                # LONG entry
                position_size = SIZE_BASE
                if atr_pct < 0.50:
                    position_size = SIZE_HIGH
                elif atr_pct > 0.70:
                    position_size = SIZE_LOW
                
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                entry_bar[i] = i
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
            elif trend == -1 and momentum_short:
                # SHORT entry
                position_size = SIZE_BASE
                if atr_pct < 0.50:
                    position_size = SIZE_HIGH
                elif atr_pct > 0.70:
                    position_size = SIZE_LOW
                
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                entry_bar[i] = i
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals