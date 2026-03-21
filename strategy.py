#!/usr/bin/env python3
"""
EXPERIMENT #089 - Regime-Adaptive MTF Ensemble with Proper HTF Alignment
==================================================================================================
Hypothesis: Recent failures (#077-#088) show complex ensembles fail due to:
1. Manual resampling causing misalignment (MUST use mtf_data helper)
2. Too many filters = too few trades + excessive churn costs
3. No regime adaptation = same strategy in all market conditions

This strategy uses:
- 4h trend via mtf_data helper (proper HTF alignment - CRITICAL)
- 15m entries with regime-adaptive logic (BBW percentile)
- 2-signal ensemble (Supertrend + RSI) instead of 7+ filters
- Discrete position sizing (0.0, ±0.25, ±0.30) to minimize churn
- Regime switch: Low BBW = trend follow, High BBW = mean revert

Why this should beat #040 and current best (Sharpe=3.653):
- Proper mtf_data alignment prevents the misalignment that killed #079, #086
- Simpler ensemble = fewer false signals = lower fees
- Regime adaptation = better performance in different vol regimes
- Conservative sizing (0.30 max) controls drawdown better than #040's 0.35
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_adaptive_mtf_ensemble_proper_htf_15m_4h_v1"
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


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        current = bbw[i]
        percentile[i] = np.sum(window <= current) / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === 15m indicators for entry timing ===
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    
    # === 4h trend filter using mtf_data helper (CRITICAL - proper HTF alignment) ===
    df_4h = get_htf_data(prices, '4h')
    if df_4h is None or len(df_4h) < 50:
        # Fallback if 4h data unavailable
        trend_4h = np.ones(n)
        st_direction_4h = np.ones(n)
        bbw_4h = np.zeros(n)
    else:
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h Supertrend for trend direction
        _, st_dir_4h_raw = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        st_direction_4h = align_htf_to_ltf(prices, df_4h, st_dir_4h_raw)
        
        # 4h BBW for regime confirmation
        _, _, _, bbw_4h_raw = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        bbw_4h = align_htf_to_ltf(prices, df_4h, bbw_4h_raw)
        
        # 4h trend: price vs SMA(50)
        sma_50_4h = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
        sma_50_aligned = align_htf_to_ltf(prices, df_4h, sma_50_4h)
        trend_4h = np.where(close > sma_50_aligned, 1, np.where(close < sma_50_aligned, -1, 0))
    
    # === Position sizing - DISCRETE levels (CRITICAL for drawdown control) ===
    SIZE_FULL = 0.30  # Slightly more conservative than #040's 0.35
    SIZE_HALF = 0.15
    
    # === Regime thresholds ===
    BBW_LOW_THRESH = 0.30  # Below 30th percentile = low vol = trend follow
    BBW_HIGH_THRESH = 0.70  # Above 70th percentile = high vol = mean revert
    
    # === Entry thresholds ===
    RSI_LONG_ENTRY = 45  # Pullback entry in uptrend
    RSI_SHORT_ENTRY = 55  # Pullback entry in downtrend
    RSI_OVERBOUGHT = 70  # Mean reversion short
    RSI_OVERSOLD = 30    # Mean reversion long
    
    # === Risk management ===
    ATR_STOP_MULT = 2.5  # Slightly wider than #040's 2.0 for less whipsaw
    MIN_ATR_PCT = 0.005  # Minimum ATR as % of price to avoid tiny stops
    
    first_valid = max(200, 100, 50 * 16)  # Need 4h data aligned (16 x 15m = 4h)
    
    # Track position state
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get current values
        price = close[i]
        atr = atr_15m[i]
        atr_pct = atr / price if price > 0 else 0
        rsi = rsi_15m[i]
        st_15m = st_direction_15m[i]
        st_4h = st_direction_4h[i]
        trend = trend_4h[i]
        bbw_pct = bbw_pct_15m[i]
        
        # Skip if ATR too small (illiquid)
        if atr_pct < MIN_ATR_PCT:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # === Check existing positions first (stoploss / take profit) ===
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
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R after TP
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
                
                # Take profit at 2R - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R after TP
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
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # === No position - check for new entries ===
        # Regime detection: BBW percentile
        is_low_vol = bbw_pct < BBW_LOW_THRESH
        is_high_vol = bbw_pct > BBW_HIGH_THRESH
        
        # 4h trend filter (must agree for trend-follow entries)
        trend_bullish = (trend == 1 and st_4h == 1)
        trend_bearish = (trend == -1 and st_4h == -1)
        
        new_signal = 0.0
        new_side = 0
        
        if is_low_vol:
            # LOW VOLATILITY REGIME: Trend-follow strategy
            # Need 4h trend confirmation + 15m Supertrend + RSI pullback
            if trend_bullish and st_15m == 1 and rsi <= RSI_LONG_ENTRY:
                new_signal = SIZE_FULL
                new_side = 1
            elif trend_bearish and st_15m == -1 and rsi >= RSI_SHORT_ENTRY:
                new_signal = -SIZE_FULL
                new_side = -1
        
        elif is_high_vol:
            # HIGH VOLATILITY REGIME: Mean-reversion strategy
            # Fade extremes when 4h trend is weak
            if abs(trend) < 1 or abs(st_4h) < 1:  # Weak 4h trend
                if rsi <= RSI_OVERSOLD and st_15m == 1:
                    new_signal = SIZE_FULL
                    new_side = 1
                elif rsi >= RSI_OVERBOUGHT and st_15m == -1:
                    new_signal = -SIZE_FULL
                    new_side = -1
        
        else:
            # MID VOLATILITY: Require stronger confirmation
            # Both 4h and 15m must agree + RSI in favor
            if trend_bullish and st_15m == 1 and rsi <= 50:
                new_signal = SIZE_FULL
                new_side = 1
            elif trend_bearish and st_15m == -1 and rsi >= 50:
                new_signal = -SIZE_FULL
                new_side = -1
        
        signals[i] = new_signal
        position_side[i] = new_side
        if new_side != 0:
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        else:
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals