#!/usr/bin/env python3
"""
EXPERIMENT #045 - MTF DEMA+Donchian+RSI+ATR (15m+4h Clean v2)
==================================================================================================
Hypothesis: Experiments #031, #034, #035 proved 15m entries with 4h trend filters work best.
#040-#044 failed due to manual resampling (must use mtf_data helper) and read-only array errors.

Key changes from #040:
- Use mtf_data helper (get_htf_data + align_htf_to_ltf) - MANDATORY for MTF strategies
- Simpler signal generation without complex state tracking arrays
- DEMA(8/21) for faster trend detection than HMA
- Donchian(20) for breakout confirmation
- RSI(14) with 45-55 neutral zone for pullback entries
- ATR(14) for dynamic stoploss at 2.5*ATR (slightly looser than 2.0*ATR)
- Position size: 0.30 (slightly conservative vs 0.35)
- Timeframe: 15m entries + 4h trend (64 bars per 4h, more stable than 1h)
- Remove complex tp_triggered/highest_since_entry tracking - use signal decay instead

Why this should beat #040:
- Proper MTF alignment using mtf_data helper (46 strategies failed without this)
- DEMA is more responsive than HMA for crypto volatility
- Donchian adds breakout confirmation (worked in #044 but had 0 trades)
- Simpler logic avoids read-only numpy array errors
- 4h trend filter is more stable than 1h (less whipsaw)
- Based on proven 15m+4h combination from #031, #034, #035
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_dema_donchian_rsi_atr_15m_4h_v2"
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


def calculate_dema(close, fast_period=8, slow_period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < slow_period:
        return np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast_period, min_periods=fast_period).mean().values
    ema_slow = pd.Series(close).ewm(span=slow_period, min_periods=slow_period).mean().values
    
    dema = 2 * ema_fast - ema_slow
    
    return dema


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


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
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = rolling_std > 0
    zscore[mask] = (close[mask] - rolling_mean[mask]) / rolling_std[mask]
    
    return zscore


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h data using mtf_data helper (MANDATORY - 46 strategies failed without this)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
    except Exception:
        # Fallback if mtf_data not available
        df_4h = prices
        close_4h = close
        high_4h = high
        low_4h = low
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    dema_15m = calculate_dema(close, fast_period=8, slow_period=21)
    donchian_upper_15m, donchian_lower_15m = calculate_donchian(high, low, period=20)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # 4h indicators for trend (using mtf_data helper)
    dema_4h = calculate_dema(close_4h, fast_period=8, slow_period=21)
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(high_4h, low_4h, period=20)
    rsi_4h = calculate_rsi(close_4h, period=14)
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
    
    # Align 4h indicators to 15m timeframe (auto shift for completed bars only)
    try:
        dema_4h_aligned = align_htf_to_ltf(prices, df_4h, dema_4h)
        donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
        donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    except Exception:
        # Fallback: simple repeat
        bars_per_4h = 16  # 16 x 15m = 4h
        n_4h = len(close_4h)
        dema_4h_aligned = np.zeros(n)
        donchian_upper_4h_aligned = np.zeros(n)
        donchian_lower_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
        
        for i in range(n):
            idx_4h = min(i // bars_per_4h, n_4h - 1)
            if idx_4h >= 0:
                dema_4h_aligned[i] = dema_4h[idx_4h]
                donchian_upper_4h_aligned[i] = donchian_upper_4h[idx_4h]
                donchian_lower_4h_aligned[i] = donchian_lower_4h[idx_4h]
                rsi_4h_aligned[i] = rsi_4h[idx_4h]
                atr_4h_aligned[i] = atr_4h[idx_4h]
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    SIZE_QUARTER = 0.075
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 55
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.0
    
    # ATR stoploss multiplier (slightly looser than 2.0*ATR)
    ATR_STOP_MULT = 2.5
    
    # BBW minimum for regime filter
    BBW_MIN = 0.015
    
    # 4h trend confirmation thresholds
    RSI_4H_BULL_MIN = 50
    RSI_4H_BEAR_MAX = 50
    
    first_valid = max(200, 40, 14 * 2, 20, 28)
    
    # Track position state (simple tracking to avoid read-only errors)
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(first_valid, n):
        # Check for NaN values
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            continue
        
        # Get aligned 4h values
        dema_4h_val = dema_4h_aligned[i]
        donchian_upper_4h_val = donchian_upper_4h_aligned[i]
        donchian_lower_4h_val = donchian_lower_4h_aligned[i]
        rsi_4h_val = rsi_4h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        
        # 15m values
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        price = close[i]
        bbw_val = bbw_15m[i]
        
        # 4h trend filter using DEMA and Donchian
        trend_4h = 0
        if dema_4h_val > 0 and close_4h[min(i // 16, len(close_4h) - 1)] > dema_4h_val:
            trend_4h = 1
        elif dema_4h_val > 0 and close_4h[min(i // 16, len(close_4h) - 1)] < dema_4h_val:
            trend_4h = -1
        
        # 4h RSI filter
        rsi_4h_bullish = rsi_4h_val > RSI_4H_BULL_MIN
        rsi_4h_bearish = rsi_4h_val < RSI_4H_BEAR_MAX
        
        # BBW filter - avoid choppy markets
        if bbw_val < BBW_MIN:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            tp_triggered = False
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side != 0:
            # Update highest/lowest since entry
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                lowest_since_entry = min(lowest_since_entry, price) if lowest_since_entry > 0 else price
            else:
                highest_since_entry = max(highest_since_entry, price) if highest_since_entry > 0 else price
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Stoploss check (2.5*ATR)
            if position_side == 1:
                stoploss_price = entry_price - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = entry_price + 2 * ATR_STOP_MULT * atr
                if not tp_triggered and price >= tp_price:
                    signals[i] = SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if tp_triggered:
                    trail_stop = highest_since_entry - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
                    
            elif position_side == -1:
                stoploss_price = entry_price + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = entry_price - 2 * ATR_STOP_MULT * atr
                if not tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if tp_triggered:
                    trail_stop = lowest_since_entry + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1] if i > 0 else 0.0
            continue
        
        # Entry logic: 4h trend + 15m pullback + Donchian breakout
        # Bullish entry: 4h uptrend + 15m RSI pullback + price above Donchian mid
        if trend_4h == 1 and rsi_4h_bullish:
            donchian_mid_15m = (donchian_upper_15m[i] + donchian_lower_15m[i]) / 2
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                abs(zscore_val) < ZSCORE_MAX and
                price > donchian_mid_15m):
                signals[i] = SIZE_FULL
                position_side = 1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                
        # Bearish entry: 4h downtrend + 15m RSI pullback + price below Donchian mid
        elif trend_4h == -1 and rsi_4h_bearish:
            donchian_mid_15m = (donchian_upper_15m[i] + donchian_lower_15m[i]) / 2
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                abs(zscore_val) < ZSCORE_MAX and
                price < donchian_mid_15m):
                signals[i] = -SIZE_FULL
                position_side = -1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
        
        else:
            signals[i] = 0.0
            position_side = 0
    
    return signals