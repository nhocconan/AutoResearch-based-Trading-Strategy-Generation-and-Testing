#!/usr/bin/env python3
"""
EXPERIMENT #018 - Donchian Trend + RSI Pullback + MACD Momentum + BBW Regime
====================================================================================
Hypothesis: Combining the proven Donchian+RSI foundation from #007 (Sharpe=4.711) with
MACD histogram momentum confirmation and Bollinger Band Width regime filtering will
improve entry timing and avoid choppy markets. This builds on the best-performing
strategy family while adding momentum confirmation.

Key innovations from #017:
- Keep 4h Donchian trend filter (proven in #007)
- Keep 1h RSI pullback entries (proven in #005, #007)
- ADD: 1h MACD histogram for momentum confirmation (must align with trend)
- ADD: Bollinger Band Width percentile to detect low-volatility regimes (cleaner breakouts)
- Position sizing: 0.35 max, discrete levels (0.0, ±0.25, ±0.35)
- Stoploss: 2*ATR trailing, Take Profit: 2R with half position reduction

Why this might beat Sharpe=5.525:
- Donchian captures trends early (better than MA crosses)
- RSI pullbacks have best entry timing historically
- MACD histogram filters out weak momentum entries
- BBW regime avoids trading during high volatility whipsaws
- Combines 4 best-performing signal types from experiment history
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_rsi_macd_bbw_v1"
timeframe = "1h"
leverage = 1.0


def calculate_donchian_channels(high, low, period=20):
    """Donchian Channels - highest high and lowest low over lookback period"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle


def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator (0-100)"""
    n = len(close)
    rsi = np.zeros(n)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    for i in range(1, n):
        if delta[i - 1] > 0:
            gain[i] = delta[i - 1]
        else:
            loss[i] = -delta[i - 1]
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """MACD - Moving Average Convergence Divergence"""
    n = len(close)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    if n < slow + signal:
        return macd_line, signal_line, histogram
    
    # Calculate EMAs
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = (close[i] - ema_fast[i - 1]) * (2 / (fast + 1)) + ema_fast[i - 1]
    
    for i in range(slow, n):
        ema_slow[i] = (close[i] - ema_slow[i - 1]) * (2 / (slow + 1)) + ema_slow[i - 1]
    
    # MACD line
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    # Signal line (EMA of MACD)
    valid_macd_start = slow - 1
    macd_valid = macd_line[valid_macd_start:]
    
    if len(macd_valid) >= signal:
        signal_line[valid_macd_start + signal - 1] = np.mean(macd_valid[:signal])
        for i in range(signal, len(macd_valid)):
            idx = valid_macd_start + i
            signal_line[idx] = (macd_line[idx] - signal_line[idx - 1]) * (2 / (signal + 1)) + signal_line[idx - 1]
    
    # Histogram
    for i in range(valid_macd_start, n):
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with Band Width calculation"""
    n = len(close)
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    band_width = np.zeros(n)
    bw_percentile = np.zeros(n)
    
    for i in range(period - 1, n):
        middle[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        band_width[i] = (upper[i] - lower[i]) / middle[i] if middle[i] > 0 else 0
    
    # Calculate BBW percentile (rolling 100-period)
    lookback = 100
    for i in range(period - 1 + lookback - 1, n):
        bw_values = band_width[i - lookback + 1:i + 1]
        bw_percentile[i] = np.sum(band_width[i - lookback + 1:i + 1] <= band_width[i]) / lookback * 100
    
    return middle, upper, lower, band_width, bw_percentile


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
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


def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator"""
    n = len(close)
    adx = np.zeros(n)
    
    if n < period * 3:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
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
    
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_di[period - 1] = 100 * np.mean(plus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    minus_di[period - 1] = 100 * np.mean(minus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100 * ((plus_dm[i] + plus_di[i - 1] * (period - 1)) / period) / atr[i] if atr[i] > 0 else 0
        minus_di[i] = 100 * ((minus_dm[i] + minus_di[i - 1] * (period - 1)) / period) / atr[i] if atr[i] > 0 else 0
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        di_diff = abs(plus_di[i] - minus_di[i])
        if di_sum > 0:
            dx[i] = 100 * di_diff / di_sum
        else:
            dx[i] = 0
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    adx_1h = calculate_adx(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_middle, bb_upper, bb_lower, bb_width, bbw_pct = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # 4h Donchian for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    donchian_upper_4h, donchian_lower_4h, donchian_mid_4h = calculate_donchian_channels(h_4h, l_4h, period=20)
    
    # 4h trend direction
    trend_4h = np.zeros(len(c_4h))
    for i in range(20, len(c_4h)):
        if c_4h[i] > donchian_mid_4h[i]:
            trend_4h[i] = 1
        elif c_4h[i] < donchian_mid_4h[i]:
            trend_4h[i] = -1
    
    # Map 4h trend back to 1h
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = min(idx_1h_to_4h[i], len(trend_4h) - 1)
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.35
    SIZE_HALF = 0.18
    
    # Entry thresholds
    RSI_LONG_ENTRY = 45
    RSI_SHORT_ENTRY = 55
    RSI_EXIT_LONG = 70
    RSI_EXIT_SHORT = 30
    
    # Filters
    ADX_MIN = 22
    BBW_PERCENTILE_MAX = 70  # Only trade when BBW is in lower 70% (avoid extreme volatility)
    ATR_STOP_MULT = 2.0
    TP_MULT = 2.0
    
    first_valid = max(100, 42, 50)
    
    entry_price = np.zeros(n)
    position_side = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(adx_1h[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi = rsi_1h[i]
        adx = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        macd_histogram = macd_hist[i]
        bbw = bbw_pct[i]
        
        # ATR filter
        if atr > 0 and atr / price > 0.06:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            continue
        
        # BBW regime filter - avoid extreme volatility
        if bbw > BBW_PERCENTILE_MAX:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ADX filter
        if adx < ADX_MIN:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Manage existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_highest = highest_since_entry[i - 1] if prev_side == 1 else 0
            prev_lowest = lowest_since_entry[i - 1] if prev_side == -1 else 0
            
            if prev_side == 1:
                current_highest = max(prev_highest, price)
                highest_since_entry[i] = current_highest
            elif prev_side == -1:
                current_lowest = min(prev_lowest, price)
                lowest_since_entry[i] = current_lowest
            
            if prev_side == 1:
                stoploss_price = max(prev_entry, prev_highest) - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    continue
                
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                if rsi > RSI_EXIT_LONG or macd_histogram < 0:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    continue
                    
            elif prev_side == -1:
                stoploss_price = min(prev_entry, prev_lowest) + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                if rsi < RSI_EXIT_SHORT or macd_histogram > 0:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            continue
        
        # New entry signals with MACD confirmation
        if trend == 1:
            if rsi <= RSI_LONG_ENTRY and rsi_1h[i - 1] > RSI_LONG_ENTRY and macd_histogram > 0:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                    
        elif trend == -1:
            if rsi >= RSI_SHORT_ENTRY and rsi_1h[i - 1] < RSI_SHORT_ENTRY and macd_histogram < 0:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals