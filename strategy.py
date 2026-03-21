#!/usr/bin/env python3
"""
EXPERIMENT #011 - MTF Donchian+MACD+RSI+Z-score+Volume (15m+1h+4h v1)
==================================================================================================
Hypothesis: Current best #004 uses Supertrend+MACD+BBW+RSI on 15m+1h+4h with Sharpe=3.653.
#040 uses HMA+Supertrend+KAMA (all trend-following) which may be redundant.

New approach for #011:
- 4h Donchian(20) for trend direction (breakout-based, different from HMA/Supertrend)
- 1h MACD histogram for momentum confirmation (different from ADX/KAMA)
- 15m RSI(14) + Z-score(20) for pullback entries (proven in #004)
- Volume spike filter for entry confirmation (new element)
- Position size: 0.35 (proven safe)
- Stoploss: 2.0*ATR (proven effective)

Why this should beat #040:
- Donchian breakout captures trend changes faster than HMA
- MACD histogram adds momentum dimension (not just trend direction)
- Volume confirmation reduces false breakouts
- Simpler trend filter (Donchian vs HMA+Supertrend+KAMA agreement)
- Based on #004's winning formula but with different trend indicator

CRITICAL: Using mtf_data helper for proper MTF alignment (no manual resampling!)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_macd_rsi_zscore_volume_15m_1h_4h_v1"
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
        else:
            zscore[i] = 0
    
    return zscore


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # EMA calculation
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = np.zeros(n)
    first_valid = slow + signal - 1
    signal_line[first_valid] = np.mean(macd_line[slow:first_valid + 1])
    
    for i in range(first_valid + 1, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands and middle)"""
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
    
    return upper, lower, middle


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume spike detection"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    volume_sma = np.zeros(n)
    
    for i in range(period - 1, n):
        volume_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return volume_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    macd_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    volume_sma_15m = calculate_volume_sma(volume, period=20)
    
    # Get 1h HTF data using mtf_data helper (CRITICAL - no manual resampling!)
    try:
        df_1h = get_htf_data(prices, '1h')
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        # 1h MACD for momentum confirmation
        macd_1h, _, macd_hist_1h = calculate_macd(close_1h, fast=12, slow=26, signal=9)
        
        # Align 1h MACD histogram to 15m timeframe
        macd_hist_1h_aligned = align_htf_to_ltf(prices, df_1h, macd_hist_1h)
    except Exception:
        # Fallback if mtf_data not available
        macd_hist_1h_aligned = macd_hist_15m
    
    # Get 4h HTF data for Donchian trend
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h Donchian for trend direction
        donchian_upper_4h, donchian_lower_4h, donchian_middle_4h = calculate_donchian(high_4h, low_4h, period=20)
        
        # Determine trend: price above middle = bullish, below = bearish
        donchian_trend_4h = np.zeros(len(close_4h))
        for i in range(20, len(close_4h)):
            if close_4h[i] > donchian_middle_4h[i]:
                donchian_trend_4h[i] = 1
            elif close_4h[i] < donchian_middle_4h[i]:
                donchian_trend_4h[i] = -1
        
        # Align 4h trend to 15m timeframe
        donchian_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_trend_4h)
    except Exception:
        # Fallback if mtf_data not available
        donchian_trend_4h_aligned = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # MACD histogram threshold for momentum confirmation
    MACD_HIST_MIN = 0.0
    
    # Volume spike multiplier (volume must be > 1.5x average)
    VOLUME_MULT = 1.5
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Minimum warmup period
    first_valid = max(200, 40, 26 + 9, 20)
    
    # Track position state
    signals = np.zeros(n)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned HTF signals
        trend_4h = donchian_trend_4h_aligned[i]
        macd_hist_1h = macd_hist_1h_aligned[i]
        
        # 15m indicators
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        macd_hist_val = macd_hist_15m[i]
        atr = atr_15m[i]
        price = close[i]
        vol = volume[i]
        vol_avg = volume_sma_15m[i]
        
        # Volume filter (avoid low volume periods)
        volume_ok = vol_avg > 0 and vol > VOLUME_MULT * vol_avg
        
        # Check stoploss and take profit for existing positions
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
            
            # Stoploss check (2.0*ATR)
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
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h Donchian trend + 1h MACD momentum + 15m RSI/Z-score pullback + Volume
        if trend_4h == 1:  # Bullish trend on 4h
            if (macd_hist_1h > MACD_HIST_MIN and  # 1h momentum positive
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and  # 15m RSI pullback
                abs(zscore_val) < ZSCORE_MAX and  # Not extreme
                volume_ok):  # Volume confirmation
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1:  # Bearish trend on 4h
            if (macd_hist_1h < -MACD_HIST_MIN and  # 1h momentum negative
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and  # 15m RSI pullback
                abs(zscore_val) < ZSCORE_MAX and  # Not extreme
                volume_ok):  # Volume confirmation
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals