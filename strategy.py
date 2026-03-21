#!/usr/bin/env python3
"""
EXPERIMENT #065 - SIMPLIFIED_ENSEMBLE_DONCHIAN_RSI_MTF_15M_4H_V1
==================================================================================================
Hypothesis: Simplified ensemble with pre-calculated indicators will avoid timeout while maintaining
edge. Key changes from #064:

1. Pre-calculate ALL indicators BEFORE main loop (vectorized) - avoids timeout
2. Replace HMA with Donchian channels (faster, proven breakout indicator)
3. Simplify position tracking - just direction and entry price
4. Keep 2/3 voting but cleaner signal definitions
5. Simpler regime detection using BBW percentile only
6. Discrete signal levels: 0.0, ±0.20, ±0.35 (cost control)

Why this should work:
- Donchian breakouts capture trends well (used in Turtle Trading)
- RSI pullbacks provide better entry timing
- 4h trend filter prevents counter-trend trades
- Pre-calculation avoids the timeout that killed #064
- Based on #055, #060, #062 which had positive Sharpe with simpler logic
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "simplified_ensemble_donchian_rsi_mtf_15m_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing - vectorized"""
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian channels - vectorized with rolling"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower


def calculate_rsi(close, period=14):
    """Calculate RSI - vectorized"""
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
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD - vectorized EMA"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    valid_start = slow + signal - 1
    if valid_start < n:
        signal_line[valid_start] = np.mean(macd_line[slow:valid_start + 1])
        for i in range(valid_start + 1, n):
            signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands - vectorized"""
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


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection - vectorized"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        percentile[i] = np.sum(window <= bbw[i]) / len(window)
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Pre-calculate ALL 15m indicators upfront (CRITICAL for speed)
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    donchian_upper_15m, donchian_middle_15m, donchian_lower_15m = calculate_donchian(high, low, period=20)
    
    # 4h trend filter using mtf_data helper
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        donchian_upper_4h, donchian_middle_4h, donchian_lower_4h = calculate_donchian(high_4h, low_4h, period=20)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        donchian_middle_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        donchian_middle_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.ones(n)
    
    # Initialize output arrays
    signals = np.zeros(n)
    position_side = np.zeros(n, dtype=np.int8)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=np.int8)
    extreme_price = np.zeros(n)
    
    # Constants
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    ATR_STOP_MULT = 2.5
    RSI_LONG_MIN, RSI_LONG_MAX = 35, 55
    RSI_SHORT_MIN, RSI_SHORT_MAX = 45, 65
    BBW_PCT_TRENDING = 0.6
    BBW_PCT_CHOPPY = 0.3
    
    first_valid = max(200, 100, 48, 26 + 9, 20, 14)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        trend_4h = 0
        if donchian_middle_4h_aligned[i] > 0:
            if close[i] > donchian_middle_4h_aligned[i]:
                trend_4h = 1
            elif close[i] < donchian_middle_4h_aligned[i]:
                trend_4h = -1
        
        # Signal 1: Donchian breakout (15m)
        signal_donchian = 0
        if close[i] > donchian_upper_15m[i - 1] if i > 0 else donchian_upper_15m[i]:
            signal_donchian = 1
        elif close[i] < donchian_lower_15m[i - 1] if i > 0 else donchian_lower_15m[i]:
            signal_donchian = -1
        
        # Signal 2: MACD momentum
        signal_macd = 0
        if macd_hist_15m[i] > 0 and macd_15m[i] > macd_signal_15m[i]:
            signal_macd = 1
        elif macd_hist_15m[i] < 0 and macd_15m[i] < macd_signal_15m[i]:
            signal_macd = -1
        
        # Signal 3: RSI pullback with trend
        signal_rsi = 0
        rsi_val = rsi_15m[i]
        if trend_4h == 1 and RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
            signal_rsi = 1
        elif trend_4h == -1 and RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
            signal_rsi = -1
        
        # Ensemble voting: need 2/3 signals agreeing
        vote_sum = signal_donchian + signal_macd + signal_rsi
        vote_count = sum([1 for s in [signal_donchian, signal_macd, signal_rsi] if s != 0])
        
        # Regime-adaptive position sizing
        bbw_pct = bbw_pct_15m[i]
        if bbw_pct >= BBW_PCT_TRENDING:
            size_mult = 1.0
        elif bbw_pct <= BBW_PCT_CHOPPY:
            size_mult = 0.5
        else:
            size_mult = 0.75
        
        # Handle existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_extreme = extreme_price[i - 1] if extreme_price[i - 1] > 0 else prev_entry
            
            # Update extreme price since entry
            if prev_side == 1:
                current_extreme = max(prev_extreme, high[i])
            else:
                current_extreme = min(prev_extreme, low[i]) if prev_extreme > 0 else low[i]
            
            extreme_price[i] = current_extreme
            atr = atr_15m[i]
            
            # Stoploss check
            exited = False
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    exited = True
            else:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    exited = True
            
            if exited:
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                extreme_price[i] = 0
                continue
            
            # Take profit check (2R) - reduce to half
            if not prev_tp:
                if prev_side == 1:
                    tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                    if close[i] >= tp_price:
                        signals[i] = SIZE_HALF * size_mult * prev_side
                        position_side[i] = prev_side
                        entry_price[i] = prev_entry
                        tp_triggered[i] = 1
                        extreme_price[i] = current_extreme
                        continue
                else:
                    tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                    if close[i] <= tp_price:
                        signals[i] = SIZE_HALF * size_mult * prev_side
                        position_side[i] = prev_side
                        entry_price[i] = prev_entry
                        tp_triggered[i] = 1
                        extreme_price[i] = current_extreme
                        continue
            
            # Trail stop at 1R profit after TP triggered
            if prev_tp:
                if prev_side == 1:
                    trail_stop = current_extreme - ATR_STOP_MULT * atr
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        extreme_price[i] = 0
                        continue
                else:
                    trail_stop = current_extreme + ATR_STOP_MULT * atr
                    if close[i] > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        extreme_price[i] = 0
                        continue
            
            # Hold position if ensemble still agrees
            if (vote_sum >= 2 and prev_side == 1) or (vote_sum <= -2 and prev_side == -1):
                signals[i] = SIZE_FULL * size_mult * prev_side if not prev_tp else SIZE_HALF * size_mult * prev_side
                position_side[i] = prev_side
                entry_price[i] = prev_entry
                tp_triggered[i] = prev_tp
                extreme_price[i] = current_extreme
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                extreme_price[i] = 0
            continue
        
        # Entry logic: need 2/3 signals agreeing
        if vote_count >= 2:
            if vote_sum >= 2:  # Bullish
                signals[i] = SIZE_FULL * size_mult
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                extreme_price[i] = close[i]
            elif vote_sum <= -2:  # Bearish
                signals[i] = -SIZE_FULL * size_mult
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                extreme_price[i] = close[i]
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals