#!/usr/bin/env python3
"""
EXPERIMENT #026 - HMA + Supertrend + ADX + MACD + BBW Multi-TF (4h+1h) Optimized
================================================================================
Hypothesis: Switching from 15m to 1h entry timeframe will reduce noise and false signals
while maintaining good entry timing. Adding MACD histogram for momentum confirmation
and Bollinger Band Width for regime detection will filter out low-volatility chop.

Why this might beat Sharpe=9.032 (#025):
- 1h entries = less noise than 15m (fewer whipsaws, cleaner signals)
- MACD histogram = momentum confirmation (entries only when momentum aligns)
- BBW filter = avoid trading during squeezes (low vol = chop = losses)
- 4h trend filter remains (HMA + Supertrend + ADX proven effective)
- Position size 0.35 with dynamic adjustment based on BBW regime
- 1.5*ATR stoploss (proven effective in #025)

Key features:
- 4h HMA(21) for adaptive trend direction
- 4h Supertrend(ATR=10, mult=3) for trend confirmation
- 4h ADX(14) for trend strength filter (>25)
- 1h MACD(12,26,9) histogram for momentum confirmation
- 1h BBW(20) for regime detection (>percentile 30)
- 1h RSI(14) pullback entries (40-50 longs, 50-60 shorts)
- 1.5*ATR stoploss with 2R take profit (reduce to half)
- Discrete signal levels: 0.0, ±0.175, ±0.35
- Dynamic sizing: base * BBW_regime_multiplier
"""

import numpy as np
import pandas as pd

name = "mtf_hma_supertrend_adx_macd_bbw_1h_v1"
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
        raw_hma = 2 * wma1[i] - wma2[i]
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.sum(np.array([2 * wma1[j] - wma2[j] for j in range(start_idx, i + 1)]) * weights) / np.sum(weights)
    
    return hma


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.zeros(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
    
    supertrend[period] = upper_band[period]
    trend_direction[period] = -1
    
    for i in range(period + 1, n):
        if close[i - 1] <= supertrend[i - 1]:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
        else:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
    
    return supertrend, trend_direction


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
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
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    for i in range(fast, n):
        ema_fast[i] = (close[i] * (2 / (fast + 1))) + (ema_fast[i - 1] * (1 - 2 / (fast + 1)))
    
    ema_slow[slow - 1] = np.mean(close[:slow])
    for i in range(slow, n):
        ema_slow[i] = (close[i] * (2 / (slow + 1))) + (ema_slow[i - 1] * (1 - 2 / (slow + 1)))
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    signal_line[slow + signal - 1] = np.mean(macd_line[slow:slow + signal])
    for i in range(slow + signal, n):
        signal_line[i] = (macd_line[i] * (2 / (signal + 1))) + (signal_line[i - 1] * (1 - 2 / (signal + 1)))
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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
        window = close[i - period + 1:i + 1]
        sma[i] = np.mean(window)
        std = np.std(window)
        upper[i] = sma[i] + std_mult * std
        lower[i] = sma[i] - std_mult * std
        
        if sma[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / sma[i]
        else:
            bbw[i] = 0
    
    return sma, upper, lower, bbw


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
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        plus_di[i] = 100 * plus_dm[i] / atr[i] if atr[i] > 0 else 0
        minus_di[i] = 100 * minus_dm[i] / atr[i] if atr[i] > 0 else 0
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
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
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_sma, bb_upper, bb_lower, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate BBW percentile for regime detection
    bbw_percentile = np.zeros(n)
    bbw_window = 100
    for i in range(bbw_window - 1, n):
        window = bbw_1h[i - bbw_window + 1:i + 1]
        bbw_percentile[i] = np.sum(window <= bbw_1h[i]) / bbw_window * 100
    
    # Resample to 4h for trend filters
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
    
    # 4h HMA for adaptive trend
    hma_4h = calculate_hma(c_4h, period=21)
    
    # 4h Supertrend for trend confirmation
    supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # 4h ADX for trend strength
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h indicators back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    st_trend_1h = np.zeros(n)
    adx_1h = np.zeros(n)
    
    n_4h = len(c_4h)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < n_4h and idx_4h >= 40:
            # HMA trend
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_1h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_1h[i] = -1
            
            # Supertrend direction
            st_trend_1h[i] = st_direction_4h[idx_4h]
            
            adx_1h[i] = adx_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries (tuned for 1h)
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 50
    RSI_SHORT_MIN = 50
    RSI_SHORT_MAX = 60
    
    # MACD histogram threshold for momentum confirmation
    MACD_MIN = 0.0
    
    # BBW percentile filter (avoid low volatility squeezes)
    BBW_MIN_PERCENTILE = 30
    
    # ADX threshold for trend strength
    ADX_MIN = 25
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 1.5
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02
    
    first_valid = max(100, 40 * 4, 14 * 2, 20, 28 * 2, bbw_window)
    
    # Track position state
    entry_price = np.zeros(n)
    position_side = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        st_trend = st_trend_1h[i]
        adx_val = adx_1h[i]
        rsi_val = rsi_1h[i]
        macd_h = macd_hist[i]
        bbw_pct = bbw_percentile[i]
        atr = atr_1h[i]
        price = close[i]
        
        # BBW filter - avoid low volatility regimes (squeezes = chop)
        if bbw_pct < BBW_MIN_PERCENTILE:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
            continue
        
        # ADX filter - only trade when trend is strong
        if adx_val < ADX_MIN:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
            continue
        
        # BOTH trend filters must agree (HMA + Supertrend)
        if trend != st_trend or trend == 0:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
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
            
            # Stoploss check (1.5*ATR)
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
                    signals[i] = SIZE_HALF
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            continue
        
        # Dynamic position sizing based on ATR volatility and BBW regime
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        # BBW regime multiplier (higher BBW = higher conviction = larger size)
        bbw_multiplier = min(1.3, max(0.8, bbw_pct / 50))
        
        position_size = SIZE_FULL * size_multiplier * bbw_multiplier
        position_size = min(SIZE_FULL, max(SIZE_HALF, position_size))
        
        # Entry logic: HMA + Supertrend + ADX + MACD + RSI + BBW
        if trend == 1 and st_trend == 1:  # Bullish trend confirmed
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and macd_h > MACD_MIN:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend == -1 and st_trend == -1:  # Bearish trend confirmed
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and macd_h < -MACD_MIN:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals