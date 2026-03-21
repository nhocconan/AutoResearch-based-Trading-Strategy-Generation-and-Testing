#!/usr/bin/env python3
"""
EXPERIMENT #110 - MTF HMA Chandelier VolRegime Enhanced (15m+4h v3)
==================================================================================================
Hypothesis: Build on #108 (Sharpe=7.706) and #109 (Sharpe=4.879) by combining their best elements:
- 4h HMA trend filter (from best performing strategies)
- 15m RSI pullback entries with tighter ranges
- ATR Chandelier exit with proper state tracking
- Volatility-adjusted position sizing using ATR percentile
- ADX + BBW regime filters to avoid chop
- Discrete signal levels (0.0, ±0.20, ±0.30) to minimize churn costs

Key improvements over #109:
- HMA instead of KAMA for smoother trend detection
- Tighter RSI ranges for better pullback entries
- Better Chandelier stop state management
- More conservative base size (0.25 max)
- Proper min_periods on all rolling calculations
- Vectorized where possible, clean loop for position tracking

Why this should beat Sharpe=16.016:
- Based on proven MTF 15m+4h structure from #096, #105, #108
- Combines trend (HMA) + momentum (RSI) + volatility (ATR/BBW) filters
- Conservative sizing prevents blowup in crypto crashes
- Chandelier exit locks in profits during strong trends
"""

import numpy as np
import pandas as pd

name = "mtf_hma_chandelier_volregime_15m_4h_v3"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
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
    
    # Calculate WMA for period, period/2, and their difference
    def wma(series, length):
        weights = np.arange(1, length + 1)
        result = np.zeros(len(series))
        for i in range(length - 1, len(series)):
            result[i] = np.sum(series[i - length + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_full = wma(close, period)
    wma_half = wma(close, half)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    diff = 2 * wma_half - wma_full
    
    hma = np.zeros(n)
    for i in range(sqrt_period - 1, n):
        weights = np.arange(1, sqrt_period + 1)
        start = i - sqrt_period + 1
        hma[i] = np.sum(diff[start:i + 1] * weights) / np.sum(weights)
    
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
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
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
    
    return zscore


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    plus_dm[1:] = np.where(
        (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
        np.maximum(0, high[1:] - high[:-1]),
        0
    )
    
    minus_dm[1:] = np.where(
        (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
        np.maximum(0, low[:-1] - low[1:]),
        0
    )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = atr > 0
    plus_di[mask] = 100 * plus_dm[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm[mask] / atr[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask = di_sum > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
    
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
    
    return upper, middle, lower, bbw


def calculate_atr_percentile(atr, lookback=100):
    """Calculate ATR percentile for volatility regime"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = atr[i - lookback + 1:i + 1]
        sorted_window = np.sort(window)
        rank = np.searchsorted(sorted_window, atr[i])
        percentile[i] = rank / lookback
    
    return percentile


def resample_to_4h(close, high, low, bars_per_4h=16):
    """Resample 15m data to 4h"""
    n = len(close)
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
    
    return c_4h, h_4h, l_4h


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.astype(float)
    high = prices["high"].values.astype(float)
    low = prices["low"].values.astype(float)
    n = len(close)
    
    signals = np.zeros(n)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=16)
    adx_15m = calculate_adx(high, low, close, period=14)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    atr_pct_15m = calculate_atr_percentile(atr_15m, lookback=100)
    
    # Resample to 4h for trend filters (16 x 15m = 4h)
    bars_per_4h = 16
    c_4h, h_4h, l_4h = resample_to_4h(close, high, low, bars_per_4h)
    n_4h = len(c_4h)
    
    # 4h indicators for trend
    hma_4h = calculate_hma(c_4h, period=16)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    atr_pct_4h = calculate_atr_percentile(atr_4h, lookback=100)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    atr_pct_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 40:
            # HMA trend
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
            atr_pct_4h_mapped[i] = atr_pct_4h[idx_4h]
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.25
    SIZE_HALF = 0.125
    
    # Volatility adjustment thresholds
    VOL_HIGH_PCT = 0.70
    VOL_LOW_PCT = 0.30
    VOL_REDUCTION = 0.5
    
    # RSI thresholds for pullback entries (tighter than #109)
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    
    # Z-score threshold
    ZSCORE_MAX = 2.0
    
    # ADX threshold for trend strength
    ADX_MIN = 18
    
    # BBW minimum for regime filter
    BBW_MIN = 0.01
    
    # Chandelier exit multiplier
    CHAN_MULT = 3.0
    
    # Minimum valid index
    first_valid = max(200, 40 * bars_per_4h, 100)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    chandelier_stop = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if atr_15m[i] == 0 or np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_4h[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_4h_val = bbw_4h_mapped[i]
        atr_4h_val = atr_4h_mapped[i]
        atr_pct = atr_pct_4h_mapped[i]
        
        # ADX filter - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # BBW filter - avoid choppy markets
        if bbw_4h_val < BBW_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filter
        if trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check exits for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
                current_high = max(prev_high, high[i])
                current_low = min(lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else low[i], low[i])
            else:
                prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
                current_high = max(highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else high[i], high[i])
                current_low = min(prev_low, low[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Update and check Chandelier stop
            if prev_side == 1:
                new_chan_stop = current_high - CHAN_MULT * atr
                chandelier_stop[i] = max(chandelier_stop[i - 1], new_chan_stop) if chandelier_stop[i - 1] > 0 else new_chan_stop
                
                # Chandelier exit stoploss
                if price < chandelier_stop[i]:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    chandelier_stop[i] = 0
                    continue
                
                # Take profit check (2R based on 4h ATR)
                tp_price = prev_entry + 2 * CHAN_MULT * atr_4h_val
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop after TP
                if prev_tp:
                    trail_stop = current_high - CHAN_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        chandelier_stop[i] = 0
                        continue
                
            elif prev_side == -1:
                new_chan_stop = current_low + CHAN_MULT * atr
                chandelier_stop[i] = min(chandelier_stop[i - 1], new_chan_stop) if chandelier_stop[i - 1] < 0 else new_chan_stop
                
                # Chandelier exit stoploss
                if price > chandelier_stop[i]:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    chandelier_stop[i] = 0
                    continue
                
                # Take profit check
                tp_price = prev_entry - 2 * CHAN_MULT * atr_4h_val
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop after TP
                if prev_tp:
                    trail_stop = current_low + CHAN_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        chandelier_stop[i] = 0
                        continue
            
            # Hold position
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            chandelier_stop[i] = chandelier_stop[i - 1]
            continue
        
        # Volatility-adjusted position sizing
        if atr_pct > VOL_HIGH_PCT:
            vol_multiplier = VOL_REDUCTION
        elif atr_pct < VOL_LOW_PCT:
            vol_multiplier = 1.0
        else:
            vol_multiplier = 1.0 - (atr_pct - VOL_LOW_PCT) / (VOL_HIGH_PCT - VOL_LOW_PCT) * (1.0 - VOL_REDUCTION)
        
        base_size = SIZE_FULL * vol_multiplier
        
        # Entry logic
        if trend == 1:  # Bullish trend
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and abs(zscore_val) < ZSCORE_MAX:
                if signals[i - 1] <= 0:  # Hysteresis
                    signals[i] = base_size
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = False
                    highest_since_entry[i] = high[i]
                    lowest_since_entry[i] = low[i]
                    chandelier_stop[i] = high[i] - CHAN_MULT * atr
                else:
                    signals[i] = signals[i - 1]
            else:
                signals[i] = 0.0
                
        elif trend == -1:  # Bearish trend
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and abs(zscore_val) < ZSCORE_MAX:
                if signals[i - 1] >= 0:  # Hysteresis
                    signals[i] = -base_size
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = False
                    highest_since_entry[i] = high[i]
                    lowest_since_entry[i] = low[i]
                    chandelier_stop[i] = low[i] + CHAN_MULT * atr
                else:
                    signals[i] = signals[i - 1]
            else:
                signals[i] = 0.0
        
        else:
            signals[i] = 0.0
    
    return signals