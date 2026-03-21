#!/usr/bin/env python3
"""
EXPERIMENT #039 - MTF Supertrend+MACD+RSI+ADX (15m+4h Optimized)
==================================================================================================
Hypothesis: Replace HMA/KAMA trend with Supertrend (proven trend follower) + MACD histogram
for momentum entry timing. Add ADX filter to ensure strong trends only.
Key changes from #038:
- Supertrend(4h, ATR=10, mult=3) for cleaner trend signals
- MACD histogram cross (1h) for momentum confirmation
- ADX(14) > 25 filter to avoid weak/choppy markets
- RSI(14) pullback within trend (45-55 range)
- Position size: 0.25 (more conservative than 0.30)
- Stoploss: 2.0*ATR with trail at 1R after TP hit

Why this could work better:
- Supertrend provides clearer trend direction with built-in ATR stop
- MACD histogram cross catches momentum shifts earlier than RSI alone
- ADX filter reduces false signals in ranging markets
- Simpler signal stack may reduce overfitting
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_macd_rsi_adx_15m_4h_v1"
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


def calculate_supertrend(high, low, close, period=10, mult=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    for i in range(period - 1, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + mult * atr[i]
        lower_band[i] = hl2 - mult * atr[i]
    
    supertrend[period - 1] = upper_band[period - 1]
    trend[period - 1] = 1
    
    for i in range(period, n):
        if close[i - 1] <= supertrend[i - 1]:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                trend[i] = -1
        else:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                trend[i] = 1
    
    return supertrend, trend


def calculate_ema(close, period=12):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = np.zeros(n)
    multiplier = 2.0 / (period + 1)
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    signal_line[slow + signal - 2] = np.mean(macd_line[slow - 1:slow + signal - 1])
    
    multiplier = 2.0 / (signal + 1)
    for i in range(slow + signal - 1, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 3:
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
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        sum_plus_dm = np.sum(plus_dm[i - period + 1:i + 1])
        sum_minus_dm = np.sum(minus_dm[i - period + 1:i + 1])
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        
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


def resample_to_higher_tf(prices, target_tf='4h'):
    """Resample to higher timeframe using open_time index - CRITICAL for no look-ahead"""
    prices_indexed = prices.set_index('open_time')
    df_resampled = prices_indexed.resample(target_tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return df_resampled


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    
    # Resample to 4h for trend filters - CRITICAL: use proper resampling
    try:
        df_4h = resample_to_higher_tf(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        n_4h = len(c_4h)
        
        # 4h Supertrend for trend direction
        _, trend_4h_raw = calculate_supertrend(h_4h, l_4h, c_4h, period=10, mult=3.0)
        
        # 4h ADX for trend strength
        adx_4h_raw = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        # CRITICAL: shift by 1 to avoid look-ahead on current forming 4h bar
        trend_4h_shifted = np.roll(trend_4h_raw, 1)
        trend_4h_shifted[0] = 0
        adx_4h_shifted = np.roll(adx_4h_raw, 1)
        adx_4h_shifted[0] = 0
        
        # Map 4h indicators back to 15m using reindex with ffill
        prices_indexed = prices.set_index('open_time')
        df_4h_indexed = df_4h.copy()
        df_4h_indexed['trend'] = trend_4h_shifted
        df_4h_indexed['adx'] = adx_4h_shifted
        
        # Reindex to 15m timestamps with forward fill
        trend_4h_mapped = df_4h_indexed['trend'].reindex(prices_indexed.index, method='ffill').values
        adx_4h_mapped = df_4h_indexed['adx'].reindex(prices_indexed.index, method='ffill').values
        
        # Handle any NaN from reindex
        trend_4h_mapped = np.nan_to_num(trend_4h_mapped, nan=0.0)
        adx_4h_mapped = np.nan_to_num(adx_4h_mapped, nan=0.0)
        
    except Exception:
        # Fallback if resampling fails
        bars_per_4h = 16
        n_4h = n // bars_per_4h
        
        c_4h = np.zeros(n_4h)
        h_4h = np.zeros(n_4h)
        l_4h = np.zeros(n_4h)
        
        for i in range(n_4h):
            start_idx = i * bars_per_4h
            end_idx = start_idx + bars_per_4h
            if end_idx <= n:
                c_4h[i] = close[end_idx - 1]
                h_4h[i] = np.max(high[start_idx:end_idx])
                l_4h[i] = np.min(low[start_idx:end_idx])
        
        _, trend_4h_raw = calculate_supertrend(h_4h, l_4h, c_4h, period=10, mult=3.0)
        adx_4h_raw = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        trend_4h_shifted = np.roll(trend_4h_raw, 1)
        trend_4h_shifted[0] = 0
        adx_4h_shifted = np.roll(adx_4h_raw, 1)
        adx_4h_shifted[0] = 0
        
        trend_4h_mapped = np.zeros(n)
        adx_4h_mapped = np.zeros(n)
        
        for i in range(n):
            idx_4h = min(i // bars_per_4h, n_4h - 1)
            if idx_4h > 0:
                trend_4h_mapped[i] = trend_4h_shifted[idx_4h]
                adx_4h_mapped[i] = adx_4h_shifted[idx_4h]
    
    # Resample to 1h for MACD momentum
    try:
        df_1h = resample_to_higher_tf(prices, '1h')
        c_1h = df_1h['close'].values
        n_1h = len(c_1h)
        
        # 1h MACD histogram for momentum
        _, _, hist_1h_raw = calculate_macd(c_1h, fast=12, slow=26, signal=9)
        
        # CRITICAL: shift by 1 to avoid look-ahead
        hist_1h_shifted = np.roll(hist_1h_raw, 1)
        hist_1h_shifted[0] = 0
        
        # Map to 15m
        prices_indexed = prices.set_index('open_time')
        df_1h_indexed = df_1h.copy()
        df_1h_indexed['macd_hist'] = hist_1h_shifted
        
        macd_1h_mapped = df_1h_indexed['macd_hist'].reindex(prices_indexed.index, method='ffill').values
        macd_1h_mapped = np.nan_to_num(macd_1h_mapped, nan=0.0)
        
    except Exception:
        bars_per_1h = 4
        n_1h = n // bars_per_1h
        
        c_1h = np.zeros(n_1h)
        for i in range(n_1h):
            start_idx = i * bars_per_1h
            end_idx = start_idx + bars_per_1h
            if end_idx <= n:
                c_1h[i] = close[end_idx - 1]
        
        _, _, hist_1h_raw = calculate_macd(c_1h, fast=12, slow=26, signal=9)
        hist_1h_shifted = np.roll(hist_1h_raw, 1)
        hist_1h_shifted[0] = 0
        
        macd_1h_mapped = np.zeros(n)
        for i in range(n):
            idx_1h = min(i // bars_per_1h, n_1h - 1)
            if idx_1h > 0:
                macd_1h_mapped[i] = hist_1h_shifted[idx_1h]
    
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels, conservative
    SIZE_FULL = 0.25
    SIZE_HALF = 0.125
    
    # Entry thresholds
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 55
    
    ZSCORE_MAX = 1.5
    ZSCORE_MIN = -1.5
    
    ADX_MIN = 25  # Minimum ADX for strong trend
    
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 40 * 16, 14 * 2, 20, 28, 42)
    
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_4h_mapped[i]
        adx = adx_4h_mapped[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        macd_hist = macd_1h_mapped[i]
        
        # ADX filter - only trade strong trends
        if adx < ADX_MIN:
            signals[i] = 0.0
            if position_side[i - 1] != 0:
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
            continue
        
        # Check if we have an existing position
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
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
                
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
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
                
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
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
            
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # New entry logic
        if trend == 1:
            # Long setup: uptrend + MACD positive + RSI pullback + Z-score neutral
            if (macd_hist > 0 and
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and
                ZSCORE_MIN <= zscore_val <= ZSCORE_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1:
            # Short setup: downtrend + MACD negative + RSI pullback + Z-score neutral
            if (macd_hist < 0 and
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and
                ZSCORE_MIN <= zscore_val <= ZSCORE_MAX):
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals