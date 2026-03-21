#!/usr/bin/env python3
"""
EXPERIMENT #004 - MTF Donchian+MACD+BBW+RSI (15m+1h+4h v2)
==================================================================================================
Hypothesis: Replace HMA trend with Donchian channel breakout (clearer trend signal) + 
MACD histogram momentum (vs pure RSI) + BBW regime filter + RSI pullback entry.

Why this should beat current best (mtf_hma_rsi_zscore_v1):
- Donchian breakout captures trend changes faster than HMA
- MACD histogram adds momentum confirmation at entry points
- BBW on 15m avoids choppy/sideways markets (reduces whipsaws)
- Three timeframes proven to work: 15m base, 1h momentum, 4h trend

Key differences from failed experiments:
- Signal size capped at 0.35 (not 1.0 like #001 which caused -87% DD)
- Proper stoploss at 2*ATR with signal→0
- Take profit at 2R (reduce to half), trail stop at 1R
- Discrete signal levels (0.0, ±0.20, ±0.35) to reduce churning costs
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_macd_bbw_rsi_15m_1h_4h_v2"
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
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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
    for i in range(n):
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands and breakout signal)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    
    return upper, middle, lower


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(n):
        if std[i] > 0:
            zscore[i] = (close[i] - mean[i]) / std[i]
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    signals = np.zeros(n)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    zscore_15m = calculate_zscore(close, period=20)
    
    # Get 1h data using mtf_data helper
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        h_1h = df_1h['high'].values
        l_1h = df_1h['low'].values
        
        macd_1h, macd_signal_1h, macd_hist_1h = calculate_macd(c_1h, fast=12, slow=26, signal=9)
        rsi_1h = calculate_rsi(c_1h, period=14)
        zscore_1h = calculate_zscore(c_1h, period=20)
        
        macd_hist_1h_aligned = align_htf_to_ltf(prices, df_1h, macd_hist_1h)
        rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
        zscore_1h_aligned = align_htf_to_ltf(prices, df_1h, zscore_1h)
    except Exception:
        macd_hist_1h_aligned = np.zeros(n)
        rsi_1h_aligned = np.zeros(n) + 50
        zscore_1h_aligned = np.zeros(n)
    
    # Get 4h data for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        donchian_upper_4h, donchian_mid_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
        
        # Donchian trend: price above middle = bullish, below = bearish
        trend_4h = np.zeros(len(c_4h))
        for i in range(len(c_4h)):
            if c_4h[i] > donchian_mid_4h[i]:
                trend_4h[i] = 1
            elif c_4h[i] < donchian_mid_4h[i]:
                trend_4h[i] = -1
        
        trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    except Exception:
        trend_4h_aligned = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # Entry thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    BBW_MIN = 0.012
    ZSCORE_MAX = 1.5
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 2, 20, 26 + 9)
    
    # Position state tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stoploss_price = 0.0
    trail_stop_price = 0.0
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        st_trend_4h = trend_4h_aligned[i] if i < len(trend_4h_aligned) else 0
        macd_hist_1h = macd_hist_1h_aligned[i] if i < len(macd_hist_1h_aligned) else 0
        rsi_1h = rsi_1h_aligned[i] if i < len(rsi_1h_aligned) else 50
        zscore_1h = zscore_1h_aligned[i] if i < len(zscore_1h_aligned) else 0
        
        price = close[i]
        
        # BBW filter - avoid choppy markets
        if bbw_15m[i] < BBW_MIN:
            if in_position:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = 0.0
            continue
        
        # Manage existing positions first
        if in_position:
            # Update highest/lowest since entry
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                lowest_since_entry = min(lowest_since_entry, price) if lowest_since_entry > 0 else price
                
                # Check stoploss
                if price < stoploss_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    continue
                
                # Check take profit (2R)
                tp_price = entry_price + 2 * ATR_STOP_MULT * atr_15m[i]
                if not tp_triggered and price >= tp_price:
                    signals[i] = SIZE_HALF
                    tp_triggered = True
                    trail_stop_price = entry_price + ATR_STOP_MULT * atr_15m[i]
                    continue
                
                # Trail stop after TP (1R profit)
                if tp_triggered:
                    new_trail = highest_since_entry - ATR_STOP_MULT * atr_15m[i]
                    trail_stop_price = max(trail_stop_price, new_trail)
                    if price < trail_stop_price:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        continue
                    
                    signals[i] = SIZE_HALF
                    continue
                
                # Hold position
                signals[i] = SIZE_FULL
                
            elif position_side == -1:
                highest_since_entry = max(highest_since_entry, price) if highest_since_entry > 0 else price
                lowest_since_entry = min(lowest_since_entry, price)
                
                # Check stoploss
                if price > stoploss_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    continue
                
                # Check take profit (2R)
                tp_price = entry_price - 2 * ATR_STOP_MULT * atr_15m[i]
                if not tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    tp_triggered = True
                    trail_stop_price = entry_price - ATR_STOP_MULT * atr_15m[i]
                    continue
                
                # Trail stop after TP (1R profit)
                if tp_triggered:
                    new_trail = lowest_since_entry + ATR_STOP_MULT * atr_15m[i]
                    trail_stop_price = min(trail_stop_price, new_trail)
                    if price > trail_stop_price:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        continue
                    
                    signals[i] = -SIZE_HALF
                    continue
                
                # Hold position
                signals[i] = -SIZE_FULL
            
            # Check if trend reversed (close position)
            if position_side == 1 and st_trend_4h == -1:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                continue
            elif position_side == -1 and st_trend_4h == 1:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                continue
            
            continue
        
        # Entry logic: 4h trend + 1h MACD momentum + 15m RSI pullback + Z-score filter
        if st_trend_4h == 1:  # Bullish trend on 4h
            if (macd_hist_1h > 0 and
                RSI_LONG_MIN <= rsi_15m[i] <= RSI_LONG_MAX and
                abs(zscore_15m[i]) < ZSCORE_MAX):
                
                signals[i] = SIZE_FULL
                in_position = True
                position_side = 1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                stoploss_price = entry_price - ATR_STOP_MULT * atr_15m[i]
                
        elif st_trend_4h == -1:  # Bearish trend on 4h
            if (macd_hist_1h < 0 and
                RSI_SHORT_MIN <= rsi_15m[i] <= RSI_SHORT_MAX and
                abs(zscore_15m[i]) < ZSCORE_MAX):
                
                signals[i] = -SIZE_FULL
                in_position = True
                position_side = -1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                stoploss_price = entry_price + ATR_STOP_MULT * atr_15m[i]
        
        else:
            signals[i] = 0.0
    
    return signals