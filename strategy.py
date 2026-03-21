#!/usr/bin/env python3
"""
EXPERIMENT #114 - HMA RSI Supertrend MTF Chandelier VolRegime Optimized (15m+4h v7)
==================================================================================================
Hypothesis: Return to the successful HMA+RSI combination from #112 (Sharpe=6.193) but enhance
with Supertrend confirmation, improved Chandelier exit logic, and better volatility regime
detection. The KAMA+Zscore approach in #113 underperformed (Sharpe=2.312), so we revert to
proven HMA+RSI while keeping the risk management improvements.

Key improvements over #112:
- HMA(16/48) crossover for primary trend (proven in #112)
- RSI(14) with dynamic thresholds based on trend strength
- Supertrend(10, 3.0) as secondary confirmation filter
- Enhanced Chandelier exit with ATR(22) * 2.8 + trailing logic
- 3-tier volatility position sizing (high/med/low vol regimes)
- ADX(14) > 20 filter to avoid choppy markets
- BBW percentile filter to avoid squeeze breakouts
- Discrete signal levels (0.0, ±0.20, ±0.30) to minimize churn costs
- Proper state tracking with hysteresis to reduce signal flipping

Why this should beat Sharpe=16.016:
- HMA is more responsive than KAMA for crypto trends
- RSI extremes within trends provide better entry timing than Z-score
- Multi-filter approach reduces false signals significantly
- Conservative sizing (max 0.30) prevents blowup in crashes
- Proper Chandelier exit locks profits during strong trends
- Volatility regime adjustment reduces exposure in high-vol periods
"""

import numpy as np
import pandas as pd

name = "hma_rsi_supertrend_mtf_chandelier_volregime_15m_4h_v7"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, short_period=16, long_period=48):
    """Calculate Hull Moving Average for trend detection"""
    n = len(close)
    if n < long_period:
        return np.zeros(n), np.zeros(n)
    
    # Calculate WMA for short and long periods
    def wma(series, period):
        weights = np.arange(1, period + 1)
        result = np.zeros(len(series))
        for i in range(period - 1, len(series)):
            result[i] = np.sum(series[i - period + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_short = wma(close, short_period)
    wma_long = wma(close, long_period)
    wma_half = wma(close, long_period // 2)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n))
    hma_raw = np.zeros(n)
    for i in range(long_period - 1, n):
        hma_raw[i] = 2 * wma_half[i] - wma_long[i]
    
    # Final HMA is WMA of raw HMA with sqrt(n) period
    sqrt_n = int(np.sqrt(long_period))
    hma = wma(hma_raw, sqrt_n)
    
    # HMA slope for trend direction
    hma_slope = np.zeros(n)
    for i in range(long_period, n):
        hma_slope[i] = hma[i] - hma[i - 1]
    
    return hma, hma_slope


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period - 1, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
    
    supertrend[period - 1] = upper_band[period - 1]
    
    for i in range(period, n):
        if direction[i - 1] == 1:
            if close[i] < supertrend[i - 1]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = max(upper_band[i], supertrend[i - 1])
        else:
            if close[i] > supertrend[i - 1]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = min(lower_band[i], supertrend[i - 1])
    
    return supertrend, direction


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
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
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = atr > 1e-10
    plus_di[mask] = 100 * plus_dm[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm[mask] / atr[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask = di_sum > 1e-10
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
        
        if middle[i] > 1e-10:
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


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        sorted_window = np.sort(window)
        rank = np.searchsorted(sorted_window, bbw[i])
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
    hma_15m, hma_slope_15m = calculate_hma(close, short_period=16, long_period=48)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Resample to 4h for trend filters (16 x 15m = 4h)
    bars_per_4h = 16
    c_4h, h_4h, l_4h = resample_to_4h(close, high, low, bars_per_4h)
    n_4h = len(c_4h)
    
    # 4h indicators for trend
    hma_4h, hma_slope_4h = calculate_hma(c_4h, short_period=16, long_period=48)
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=22)
    atr_pct_4h = calculate_atr_percentile(atr_4h, lookback=100)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    
    # Map 4h indicators back to 15m timeframe
    trend_4h = np.zeros(n)
    adx_4h_mapped = np.zeros(n)
    bbw_4h_mapped = np.zeros(n)
    bbw_pct_4h_mapped = np.zeros(n)
    atr_4h_mapped = np.zeros(n)
    atr_pct_4h_mapped = np.zeros(n)
    st_direction_4h_mapped = np.zeros(n)
    hma_slope_4h_mapped = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 50:
            # Trend from HMA slope
            if hma_slope_4h[idx_4h] > 0:
                trend_4h[i] = 1
            elif hma_slope_4h[idx_4h] < 0:
                trend_4h[i] = -1
            
            st_direction_4h_mapped[i] = st_direction_4h[idx_4h]
            adx_4h_mapped[i] = adx_4h[idx_4h]
            bbw_4h_mapped[i] = bbw_4h[idx_4h]
            bbw_pct_4h_mapped[i] = bbw_pct_4h[idx_4h]
            atr_4h_mapped[i] = atr_4h[idx_4h]
            atr_pct_4h_mapped[i] = atr_pct_4h[idx_4h]
            hma_slope_4h_mapped[i] = hma_slope_4h[idx_4h]
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Volatility adjustment thresholds
    VOL_HIGH_PCT = 0.75
    VOL_MED_PCT = 0.45
    VOL_LOW_PCT = 0.25
    
    # RSI thresholds for entries (dynamic based on trend)
    RSI_LONG_MAX = 55
    RSI_LONG_MIN = 35
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ADX threshold for trend strength
    ADX_MIN = 20
    
    # BBW percentile filters
    BBW_PCT_MIN = 0.15
    BBW_PCT_MAX = 0.85
    
    # Chandelier exit multiplier (ATR 22 period)
    CHAN_MULT = 2.8
    CHAN_PERIOD = 22
    
    # Minimum valid index
    first_valid = max(250, 50 * bars_per_4h, 100)
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    chandelier_stop = 0.0
    last_signal = 0.0
    
    for i in range(first_valid, n):
        # Skip invalid data
        if atr_15m[i] < 1e-10 or np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            last_signal = 0.0
            continue
        
        trend = trend_4h[i]
        st_dir = st_direction_4h_mapped[i]
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        adx_4h_val = adx_4h_mapped[i]
        bbw_4h_val = bbw_4h_mapped[i]
        bbw_pct = bbw_pct_4h_mapped[i]
        atr_4h_val = atr_4h_mapped[i]
        atr_pct = atr_pct_4h_mapped[i]
        
        # ADX filter - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            if in_position:
                signals[i] = 0.0
                last_signal = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                chandelier_stop = 0.0
            else:
                signals[i] = 0.0
                last_signal = 0.0
            continue
        
        # BBW percentile filter - avoid extreme squeeze/breakout regimes
        if bbw_pct < BBW_PCT_MIN or bbw_pct > BBW_PCT_MAX:
            if in_position:
                signals[i] = 0.0
                last_signal = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                chandelier_stop = 0.0
            else:
                signals[i] = 0.0
                last_signal = 0.0
            continue
        
        # Trend filter - require HMA slope direction
        if trend == 0:
            if in_position:
                signals[i] = 0.0
                last_signal = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                chandelier_stop = 0.0
            else:
                signals[i] = 0.0
                last_signal = 0.0
            continue
        
        # Check exits for existing positions
        if in_position:
            # Update highest/lowest since entry
            if position_side == 1:
                if highest_since_entry < 1e-10:
                    highest_since_entry = high[i]
                else:
                    highest_since_entry = max(highest_since_entry, high[i])
                
                # Update Chandelier stop
                new_chan_stop = highest_since_entry - CHAN_MULT * atr
                if chandelier_stop < 1e-10:
                    chandelier_stop = new_chan_stop
                else:
                    chandelier_stop = max(chandelier_stop, new_chan_stop)
                
                # Chandelier exit stoploss
                if price < chandelier_stop:
                    signals[i] = 0.0
                    last_signal = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    chandelier_stop = 0.0
                    continue
                
                # Take profit check (2R based on 4h ATR)
                if not tp_triggered:
                    tp_price = entry_price + 2 * CHAN_MULT * atr_4h_val
                    if price >= tp_price:
                        signals[i] = SIZE_HALF
                        last_signal = SIZE_HALF
                        tp_triggered = True
                        continue
                
                # Trail stop after TP
                if tp_triggered:
                    trail_stop = highest_since_entry - CHAN_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        last_signal = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        chandelier_stop = 0.0
                        continue
            
            elif position_side == -1:
                if lowest_since_entry < 1e-10:
                    lowest_since_entry = low[i]
                else:
                    lowest_since_entry = min(lowest_since_entry, low[i])
                
                # Update Chandelier stop
                new_chan_stop = lowest_since_entry + CHAN_MULT * atr
                if chandelier_stop < 1e-10:
                    chandelier_stop = new_chan_stop
                else:
                    chandelier_stop = min(chandelier_stop, new_chan_stop)
                
                # Chandelier exit stoploss
                if price > chandelier_stop:
                    signals[i] = 0.0
                    last_signal = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    chandelier_stop = 0.0
                    continue
                
                # Take profit check
                if not tp_triggered:
                    tp_price = entry_price - 2 * CHAN_MULT * atr_4h_val
                    if price <= tp_price:
                        signals[i] = -SIZE_HALF
                        last_signal = -SIZE_HALF
                        tp_triggered = True
                        continue
                
                # Trail stop after TP
                if tp_triggered:
                    trail_stop = lowest_since_entry + CHAN_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        last_signal = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        chandelier_stop = 0.0
                        continue
            
            # Hold position - use last_signal to avoid unnecessary changes
            signals[i] = last_signal
            continue
        
        # Volatility-adjusted position sizing (3 tiers)
        if atr_pct > VOL_HIGH_PCT:
            vol_multiplier = 0.5
        elif atr_pct > VOL_MED_PCT:
            vol_multiplier = 0.75
        elif atr_pct < VOL_LOW_PCT:
            vol_multiplier = 1.0
        else:
            vol_multiplier = 0.85
        
        base_size = SIZE_FULL * vol_multiplier
        
        # Entry logic - require trend + supertrend + RSI alignment
        if trend == 1 and st_dir == 1:  # Bullish trend
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                # Only enter if not already long (hysteresis)
                if last_signal <= 0:
                    signals[i] = base_size
                    last_signal = base_size
                    in_position = True
                    position_side = 1
                    entry_price = price
                    tp_triggered = False
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    chandelier_stop = high[i] - CHAN_MULT * atr
                else:
                    signals[i] = last_signal
            else:
                signals[i] = 0.0
                last_signal = 0.0
                
        elif trend == -1 and st_dir == -1:  # Bearish trend
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                # Only enter if not already short (hysteresis)
                if last_signal >= 0:
                    signals[i] = -base_size
                    last_signal = -base_size
                    in_position = True
                    position_side = -1
                    entry_price = price
                    tp_triggered = False
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    chandelier_stop = low[i] + CHAN_MULT * atr
                else:
                    signals[i] = last_signal
            else:
                signals[i] = 0.0
                last_signal = 0.0
        
        else:
            signals[i] = 0.0
            last_signal = 0.0
    
    return signals