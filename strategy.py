#!/usr/bin/env python3
"""
EXPERIMENT #022 - DEMA Bollinger Mean Reversion with MTF Trend Filter (1h Primary)
==================================================================================================
Hypothesis: Current best (Sharpe=0.537) uses 4h HMA+RSI pullback. This uses 1h DEMA+Bollinger
for more frequent entries while maintaining quality via 4h+1d trend filters.

Key innovations:
1. 1h PRIMARY + 4h/1d HTF: More trades than 4h, cleaner than 15m/30m
2. DEMA for faster trend detection than HMA (Double EMA reduces lag further)
3. Bollinger Band position: Enter when price near lower band in uptrend (mean reversion)
4. RSI confirmation: Avoid entering at extremes (>70 or <30)
5. ATR-based dynamic sizing: size = base * (target_ATR / current_ATR)
6. Tighter stoploss: 1.5*ATR vs 2.0*ATR (faster exits on wrong trades)

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- 1h timeframe = 4x more potential trades than 4h
- DEMA faster response than HMA = catch trends earlier
- Bollinger position filter = better entry timing (buy low in uptrend)
- Dynamic sizing = reduce position when volatility spikes (protects during crashes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "dema_bb_mean_reversion_mtf_1h_v1"
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


def calculate_dema(close, period=21):
    """
    Double Exponential Moving Average - faster than EMA, less lag
    DEMA = 2*EMA(n) - EMA(EMA(n))
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    
    dema = 2 * ema1 - ema2
    return dema.values


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    close_series = pd.Series(close)
    middle = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    return upper, middle, lower


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    if n < len(atr) or len(atr) == 0:
        return np.zeros(n), np.zeros(n)
    
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        if atr[i] == 0:
            continue
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
    
    first_valid = np.where(atr > 0)[0]
    if len(first_valid) == 0:
        return supertrend, trend
    
    start_idx = first_valid[0]
    supertrend[start_idx] = upper_band[start_idx]
    trend[start_idx] = 1
    
    for i in range(start_idx + 1, n):
        if atr[i] == 0:
            supertrend[i] = supertrend[i - 1]
            trend[i] = trend[i - 1]
            continue
        
        if trend[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = max(supertrend[i - 1], lower_band[i])
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = min(supertrend[i - 1], upper_band[i])
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (PRIMARY TIMEFRAME) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    dema_1h_fast = calculate_dema(close, period=8)
    dema_1h_slow = calculate_dema(close, period=21)
    bb_upper_1h, bb_middle_1h, bb_lower_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    supertrend_1h, st_trend_1h = calculate_supertrend(high, low, close, atr_1h, multiplier=3.0)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        dema_4h = calculate_dema(close_4h, period=21)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        _, st_trend_4h = calculate_supertrend(high_4h, low_4h, close_4h, atr_4h, multiplier=3.0)
        
        dema_4h_aligned = align_htf_to_ltf(prices, df_4h, dema_4h)
        st_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, st_trend_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        dema_4h_aligned = np.zeros(n)
        st_trend_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
    
    # ========== 1d INDICATORS (REGIME FILTER) - PROPER MTF ==========
    try:
        df_1d = get_htf_data(prices, '1d')
        close_1d = df_1d['close'].values
        
        dema_1d = calculate_dema(close_1d, period=21)
        
        dema_1d_aligned = align_htf_to_ltf(prices, df_1d, dema_1d)
        
    except Exception:
        dema_1d_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DYNAMIC based on ATR
    SIZE_BASE = 0.20
    SIZE_HIGH = 0.30
    TARGET_ATR_PCT = 0.02  # Target 2% ATR
    
    # Stoploss - TIGHTER than baseline
    ATR_STOP_MULT = 1.5
    
    # RSI filter zones
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # Bollinger Band position thresholds
    BB_LONG_THRESHOLD = 0.15  # Price within 15% of lower band
    BB_SHORT_THRESHOLD = 0.15  # Price within 15% of upper band
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_lower_1h[i]) or np.isnan(bb_upper_1h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi_val = rsi_1h[i]
        st_trend_val = st_trend_1h[i]
        dema_fast = dema_1h_fast[i]
        dema_slow = dema_1h_slow[i]
        
        # Bollinger Band position
        bb_range = bb_upper_1h[i] - bb_lower_1h[i]
        if bb_range > 0:
            bb_position = (price - bb_lower_1h[i]) / bb_range  # 0=lower, 1=upper
        else:
            bb_position = 0.5
        
        # 4h trend filters
        dema_4h_val = dema_4h_aligned[i]
        st_trend_4h_val = st_trend_4h_aligned[i]
        
        # 1d regime filter
        dema_1d_val = dema_1d_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if dema_4h_val > 0 and price > dema_4h_val:
            trend_4h = 1
        elif dema_4h_val > 0 and price < dema_4h_val:
            trend_4h = -1
        
        if st_trend_4h_val == 1:
            trend_4h = max(trend_4h, 1)
        elif st_trend_4h_val == -1:
            trend_4h = min(trend_4h, -1)
        
        # Determine 1d regime
        regime_1d = 0
        if dema_1d_val > 0 and price > dema_1d_val:
            regime_1d = 1
        elif dema_1d_val > 0 and price < dema_1d_val:
            regime_1d = -1
        
        # ========== CHECK EXISTING POSITIONS ==========
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
            
            # Stoploss check (1.5*ATR - tighter)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    # Dynamic sizing for TP
                    current_atr_pct = atr / prev_entry if prev_entry > 0 else 0.02
                    size_mult = min(2.0, TARGET_ATR_PCT / current_atr_pct) if current_atr_pct > 0 else 1.0
                    signals[i] = SIZE_BASE * size_mult
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    current_atr_pct = atr / prev_entry if prev_entry > 0 else 0.02
                    size_mult = min(2.0, TARGET_ATR_PCT / current_atr_pct) if current_atr_pct > 0 else 1.0
                    signals[i] = -SIZE_BASE * size_mult
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
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
        
        # ========== ENTRY LOGIC - BOLLINGER MEAN REVERSION IN TREND ==========
        # Dynamic position sizing based on current ATR
        current_atr_pct = atr / price if price > 0 else 0.02
        size_mult = min(2.0, TARGET_ATR_PCT / current_atr_pct) if current_atr_pct > 0 else 1.0
        
        # LONG: 4h trend up + 1d regime up + Price near BB lower + RSI not oversold + DEMA fast > slow
        long_condition = (
            trend_4h == 1 and
            regime_1d >= 0 and
            bb_position <= BB_LONG_THRESHOLD and
            rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX and
            dema_fast > dema_slow and
            st_trend_val == 1
        )
        
        # SHORT: 4h trend down + 1d regime down + Price near BB upper + RSI not overbought + DEMA fast < slow
        short_condition = (
            trend_4h == -1 and
            regime_1d <= 0 and
            bb_position >= (1 - BB_SHORT_THRESHOLD) and
            rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX and
            dema_fast < dema_slow and
            st_trend_val == -1
        )
        
        # High conviction: 1d regime aligns with 4h trend
        high_conviction_long = long_condition and regime_1d == 1
        high_conviction_short = short_condition and regime_1d == -1
        
        if long_condition:
            size = min(SIZE_HIGH * size_mult, 0.40) if high_conviction_long else min(SIZE_BASE * size_mult, 0.30)
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            size = min(SIZE_HIGH * size_mult, 0.40) if high_conviction_short else min(SIZE_BASE * size_mult, 0.30)
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals