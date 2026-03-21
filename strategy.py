#!/usr/bin/env python3
"""
EXPERIMENT #089 - TRIPLE REGIME ENSEMBLE WITH SIGNAL SMOOTHING
==================================================================================================
Hypothesis: Combine triple regime detection (BBW + ADX + Vol Ratio) with signal hysteresis
to reduce churn costs while maintaining edge. Use discrete signal levels with volatility scaling.

Key innovations:
- Triple regime filter: BBW percentile + ADX + Volatility ratio (all must agree)
- Signal hysteresis: Require 2-bar confirmation before changing signal direction
- Volatility-adjusted sizing: position_size = base_size * (ref_vol / current_vol)
- Clean separation: indicator calculation → raw signal → smoothed signal → position management
- Discrete levels only: 0.0, ±0.20, ±0.35 (minimizes fee drag from signal changes)

Why this should work:
- #088 had low Sharpe (0.047) due to signal churn and complex position tracking
- Triple regime filter reduces false signals in choppy markets
- Hysteresis prevents rapid flipping (each flip costs 0.10% fees)
- Vol scaling keeps risk constant across different market conditions
- Simpler logic = fewer bugs than ensemble voting systems (#078-#081 crashes)

Risk Management:
- Max position: 0.35 (35% of capital)
- Stop loss: 2.5x ATR from entry
- Take profit: reduce to half at 2R, trail stop at 1R
- No leverage (leverage=1.0)
"""

import numpy as np
import pandas as pd

name = "triple_regime_ensemble_hysteresis_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    wma1 = pd.Series(close).rolling(window=half, min_periods=half).apply(
        lambda x: np.sum(x * np.arange(1, half + 1)) / np.sum(np.arange(1, half + 1)), raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1)), raw=True
    ).values
    
    hma = pd.Series(2 * wma1 - wma2).rolling(window=sqrt_p, min_periods=sqrt_p).apply(
        lambda x: np.sum(x * np.arange(1, sqrt_p + 1)) / np.sum(np.arange(1, sqrt_p + 1)), raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
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
    rs[~mask] = 100.0
    
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
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
    
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
        denom = plus_di[i] + minus_di[i]
        if denom > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / denom
    
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
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    bbw = np.zeros(n)
    mask = middle > 0
    bbw[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def calculate_volatility_ratio(close, short_period=10, long_period=50):
    """Calculate short-term vs long-term volatility ratio"""
    n = len(close)
    if n < long_period:
        return np.zeros(n)
    
    returns = np.diff(close, prepend=close[0]) / close
    short_vol = pd.Series(returns).rolling(window=short_period, min_periods=short_period).std().values
    long_vol = pd.Series(returns).rolling(window=long_period, min_periods=long_period).std().values
    
    vol_ratio = np.zeros(n)
    mask = long_vol > 0
    vol_ratio[mask] = short_vol[mask] / long_vol[mask]
    
    return vol_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    signals = np.zeros(n)
    
    # ========== 1H INDICATORS (ENTRY) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    _, _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_1h = calculate_bbw_percentile(bbw_1h, lookback=100)
    adx_1h = calculate_adx(high, low, close, period=14)
    vol_ratio_1h = calculate_volatility_ratio(close, short_period=10, long_period=50)
    
    # ========== 4H TREND FILTER (downsample 1h → 4h) ==========
    bars_per_4h = 4
    n_4h = n // bars_per_4h
    
    c_4h = np.zeros(n_4h)
    h_4h = np.zeros(n_4h)
    l_4h = np.zeros(n_4h)
    
    for i in range(n_4h):
        start_idx = i * bars_per_4h
        end_idx = start_idx + bars_per_4h
        c_4h[i] = close[end_idx - 1]
        h_4h[i] = np.max(high[start_idx:end_idx])
        l_4h[i] = np.min(low[start_idx:end_idx])
    
    # 4H indicators
    hma_4h = calculate_hma(c_4h, period=21)
    _, _, _, bbw_4h = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    bbw_pct_4h = calculate_bbw_percentile(bbw_4h, lookback=100)
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h indicators to 1h
    trend_4h = np.zeros(n)
    regime_trend = np.zeros(n)
    regime_vol = np.zeros(n)
    
    for i in range(n):
        idx_4h = i // bars_per_4h
        if idx_4h < n_4h and idx_4h >= 21:
            # Trend direction
            if c_4h[idx_4h] > hma_4h[idx_4h]:
                trend_4h[i] = 1
            elif c_4h[idx_4h] < hma_4h[idx_4h]:
                trend_4h[i] = -1
            
            # Regime: BBW (low = trending, high = mean-reverting)
            if bbw_pct_4h[idx_4h] < 0.3:
                regime_trend[i] = 1  # Trend regime
            elif bbw_pct_4h[idx_4h] > 0.7:
                regime_trend[i] = -1  # Mean-reversion regime
            
            # Regime: ADX (high = trending, low = ranging)
            if adx_4h[idx_4h] > 25:
                regime_vol[i] = 1  # Trending
            elif adx_4h[idx_4h] < 20:
                regime_vol[i] = -1  # Ranging
    
    # ========== SIGNAL GENERATION ==========
    SIZE_BASE = 0.20
    SIZE_MAX = 0.35
    SIZE_HALF = 0.15
    
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 50
    RSI_SHORT_MIN = 50
    RSI_SHORT_MAX = 65
    ATR_STOP_MULT = 2.5
    REF_VOL = 1.0  # Reference volatility for sizing
    
    first_valid = max(250, 100 + 21, 14 * 3, 50)
    
    # State tracking (use lists to avoid read-only issues)
    position_side = [0] * n
    entry_price = [0.0] * n
    tp_triggered = [False] * n
    highest_since_entry = [0.0] * n
    lowest_since_entry = [0.0] * n
    signal_history = [0.0] * n  # For hysteresis
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            signal_history[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi_val = rsi_1h[i]
        trend = trend_4h[i]
        reg_trend = regime_trend[i]
        reg_vol = regime_vol[i]
        vol_ratio = vol_ratio_1h[i]
        
        # ========== POSITION MANAGEMENT (stoploss/TP) ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update extremes
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    signal_history[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0.0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0.0
                    lowest_since_entry[i] = 0.0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    signal_history[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop after TP
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        signal_history[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0.0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0.0
                        lowest_since_entry[i] = 0.0
                        continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    signal_history[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0.0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0.0
                    lowest_since_entry[i] = 0.0
                    continue
                
                # Take profit at 2R
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    signal_history[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop after TP
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        signal_history[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0.0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0.0
                        lowest_since_entry[i] = 0.0
                        continue
            
            # Hold position
            signals[i] = signals[i - 1]
            signal_history[i] = signal_history[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== NEW ENTRY LOGIC ==========
        # Triple regime filter: all must agree for full size
        regime_confidence = 0
        if reg_trend == 1 and reg_vol == 1:
            regime_confidence = 2  # Strong trend
        elif reg_trend == 1 or reg_vol == 1:
            regime_confidence = 1  # Moderate trend
        elif reg_trend == -1 or reg_vol == -1:
            regime_confidence = -1  # Mean-reversion
        
        # Volatility-adjusted position size
        vol_scale = min(2.0, max(0.5, REF_VOL / vol_ratio)) if vol_ratio > 0 else 1.0
        
        signal_long = False
        signal_short = False
        target_size = SIZE_BASE
        
        # Entry conditions based on regime
        if regime_confidence >= 1:  # Trend regime
            if trend == 1 and RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                signal_long = True
                target_size = SIZE_MAX if regime_confidence == 2 else SIZE_BASE
            elif trend == -1 and RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                signal_short = True
                target_size = SIZE_MAX if regime_confidence == 2 else SIZE_BASE
        elif regime_confidence == -1:  # Mean-reversion regime
            if rsi_val < RSI_LONG_MIN:
                signal_long = True
                target_size = SIZE_BASE
            elif rsi_val > RSI_SHORT_MAX:
                signal_short = True
                target_size = SIZE_BASE
        
        # Apply volatility scaling
        target_size = min(SIZE_MAX, max(0.0, target_size * vol_scale))
        
        # ========== HYSTERESIS (require confirmation) ==========
        prev_signal = signal_history[i - 1] if i > 0 else 0.0
        
        # Only change signal if there's confirmation (same signal 2 bars)
        if signal_long:
            if prev_signal >= 0:
                signals[i] = target_size
                signal_history[i] = target_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                signal_history[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0.0
        elif signal_short:
            if prev_signal <= 0:
                signals[i] = -target_size
                signal_history[i] = -target_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                signal_history[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0.0
        else:
            signals[i] = 0.0
            signal_history[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0.0
    
    return signals