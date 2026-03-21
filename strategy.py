#!/usr/bin/env python3
"""
EXPERIMENT #050 - MTF Supertrend+HMA+RSI+Z-score+BBW (15m+4h Clean v3)
==================================================================================================
Hypothesis: Current best (Sharpe=3.653) uses 15m+1h+4h. This experiment simplifies to 15m+4h
with cleaner MTF logic using mtf_data helper (MANDATORY per rules).

Key changes from #040:
- Use mtf_data helper (get_htf_data, align_htf_to_ltf) instead of manual resampling
- Timeframe: 15m entries + 4h trend (simpler than 15m+1h+4h)
- Position size: 0.30 (slightly more conservative than 0.35)
- Stoploss: 2.5*ATR (wider than #040's 2.0*ATR for less whipsaw)
- Simplified state tracking to avoid read-only errors
- ADX threshold: 20 (lower than #040's 25 for more trades)
- RSI thresholds: 35-65 (wider range for more entry opportunities)

Why this should beat #040:
- Proper MTF alignment using mtf_data helper (46 strategies failed without this)
- 4h trend filter is more stable than 1h (less noise)
- Wider RSI range captures more pullback entries
- Simpler code = fewer bugs (no read-only assignment errors)
- Based on proven winning combination from current best (Supertrend+HMA+RSI)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_hma_rsi_zscore_bbw_15m_4h_v3"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    hma = pd.Series(2 * wma1 - wma2).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return np.nan_to_num(rsi, nan=50.0)


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
    
    return np.nan_to_num(zscore, nan=0.0)


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    mid = (high + low) / 2
    upper_band = mid + multiplier * atr
    lower_band = mid - multiplier * atr
    
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
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return np.nan_to_num(adx, nan=0.0)


def calculate_bbw(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = rolling_mean + std_mult * rolling_std
    lower = rolling_mean - std_mult * rolling_std
    
    bbw = np.zeros(n)
    for i in range(n):
        if rolling_mean[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / rolling_mean[i]
    
    return np.nan_to_num(bbw, nan=0.0)


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    bbw_15m = calculate_bbw(close, period=20, std_mult=2.0)
    
    # Get 4h data using mtf_data helper (MANDATORY - no manual resampling)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h indicators for trend direction
    close_4h = df_4h["close"].values
    high_4h = df_4h["high"].values
    low_4h = df_4h["low"].values
    
    hma_4h = calculate_hma(close_4h, period=21)
    supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
    bbw_4h = calculate_bbw(close_4h, period=20, std_mult=2.0)
    
    # Align 4h indicators to 15m timeframe (auto shift for completed bars only)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    bbw_4h_aligned = align_htf_to_ltf(prices, df_4h, bbw_4h)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries (wider range for more opportunities)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.5
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 20
    
    # BBW minimum for regime filter
    BBW_MIN = 0.015
    
    # ATR stoploss multiplier (wider than #040 for less whipsaw)
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 14 * 2, 20, 28)
    
    # Track position state
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(first_valid, n):
        # Check for NaN/invalid values
        if (np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or 
            atr_15m[i] == 0 or np.isnan(hma_4h_aligned[i]) or np.isnan(st_4h_aligned[i])):
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            tp_triggered = False
            continue
        
        # 4h trend filters
        trend_4h = 1 if close[i] > hma_4h_aligned[i] else -1
        st_trend_4h = st_4h_aligned[i]
        adx_4h_val = adx_4h_aligned[i]
        bbw_4h_val = bbw_4h_aligned[i]
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
            continue
        
        # BBW filter - avoid choppy markets (4h)
        if bbw_4h_val < BBW_MIN:
            signals[i] = 0.0
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        
        # Check stoploss and take profit for existing positions
        if position_side != 0:
            prev_side = position_side
            prev_entry = entry_price
            prev_tp = tp_triggered
            prev_high = highest_since_entry if highest_since_entry > 0 else prev_entry
            prev_low = lowest_since_entry if lowest_since_entry > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry = current_high
            lowest_since_entry = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Entry logic: 4h HMA + Supertrend + ADX + BBW + 15m RSI + Z-score
        if trend_4h == 1 and st_trend_4h == 1:  # Bullish trend confirmed on 4h
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                abs(zscore_val) < ZSCORE_MAX):  # Pullback + not extreme
                signals[i] = SIZE_FULL
                position_side = 1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                
        elif trend_4h == -1 and st_trend_4h == -1:  # Bearish trend confirmed on 4h
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                abs(zscore_val) < ZSCORE_MAX):  # Pullback + not extreme
                signals[i] = -SIZE_FULL
                position_side = -1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
        
        else:
            signals[i] = 0.0
    
    return signals