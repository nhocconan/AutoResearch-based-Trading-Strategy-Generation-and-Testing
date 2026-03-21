#!/usr/bin/env python3
"""
EXPERIMENT #046 - MTF KAMA+RSI+ATR (1h+4h Clean v1)
==================================================================================================
Hypothesis: After 45 experiments, 15m+4h has been heavily tested. Let's try 1h+4h combination
which should have fewer whipsaws than 15m while still catching major moves. 

Key changes from failed #045:
- Fix Donchian calculation bug (close was not defined in function scope)
- Use 1h entries + 4h trend (different from 15m+4h which has been tried 10+ times)
- KAMA (Kaufman Adaptive MA) instead of DEMA/HMA - adapts to volatility regimes
- Simpler signal generation - compute signals vectorized, then apply stoploss logic
- Position size: 0.30 (conservative, proven range)
- ATR stoploss: 2.0*ATR (tighter than 2.5*ATR to protect capital)
- Add volume confirmation filter (volume > SMA20 of volume)

Why this should work:
- 1h timeframe has fewer false signals than 15m (less noise)
- KAMA adapts to crypto volatility better than fixed EMA/HMA
- Simpler logic avoids read-only numpy array errors that crashed #034, #040-#045
- 4h trend filter provides regime context (proven in #031, #034, #035)
- Volume filter avoids low-liquidity traps

Based on lessons from 45 experiments:
- MTF MUST use mtf_data helper (46 strategies failed without this)
- Signal magnitude max 0.35 (BTC crashed 77% in 2022)
- Discrete signal levels to avoid fee churning (0.0, ±0.15, ±0.30)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_atr_1h_4h_v1"
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


def calculate_kama(close, period=10, fast_sc=2/11, slow_sc=2/31):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - moves fast in trending markets, slow in choppy
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period - 1, n):
        price_change = abs(close[i] - close[i - period + 1])
        volatility = np.sum(np.abs(np.diff(close[i - period + 1:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    sc = np.zeros(n)
    for i in range(period - 1, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    # Calculate KAMA
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


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


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # Get 4h data using mtf_data helper (MANDATORY - 46 strategies failed without this)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        volume_4h = df_4h['volume'].values if "volume" in df_4h.columns else np.ones(len(close_4h))
    except Exception:
        # Fallback if mtf_data not available
        df_4h = prices
        close_4h = close
        high_4h = high
        low_4h = low
        volume_4h = volume
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, period=10)
    donchian_upper_1h, donchian_lower_1h = calculate_donchian(high, low, period=20)
    _, _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    volume_sma_1h = calculate_volume_sma(volume, period=20)
    
    # 4h indicators for trend (using mtf_data helper)
    kama_4h = calculate_kama(close_4h, period=10)
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(high_4h, low_4h, period=20)
    rsi_4h = calculate_rsi(close_4h, period=14)
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
    
    # Align 4h indicators to 1h timeframe (auto shift for completed bars only)
    try:
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
        donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    except Exception:
        # Fallback: simple repeat
        bars_per_4h = 4  # 4 x 1h = 4h
        n_4h = len(close_4h)
        kama_4h_aligned = np.zeros(n)
        donchian_upper_4h_aligned = np.zeros(n)
        donchian_lower_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
        
        for i in range(n):
            idx_4h = min(i // bars_per_4h, n_4h - 1)
            if idx_4h >= 0:
                kama_4h_aligned[i] = kama_4h[idx_4h]
                donchian_upper_4h_aligned[i] = donchian_upper_4h[idx_4h]
                donchian_lower_4h_aligned[i] = donchian_lower_4h[idx_4h]
                rsi_4h_aligned[i] = rsi_4h[idx_4h]
                atr_4h_aligned[i] = atr_4h[idx_4h]
    
    # Generate raw signals first (vectorized)
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # 4h trend confirmation thresholds
    RSI_4H_BULL_MIN = 50
    RSI_4H_BEAR_MAX = 50
    
    # BBW minimum for regime filter
    BBW_MIN = 0.01
    
    # Volume filter threshold
    VOLUME_MULT = 1.0
    
    first_valid = max(200, 40, 14 * 2, 20, 28)
    
    # Compute raw entry signals
    for i in range(first_valid, n):
        # Check for NaN/zero values
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            continue
        
        # Get aligned 4h values
        kama_4h_val = kama_4h_aligned[i]
        donchian_upper_4h_val = donchian_upper_4h_aligned[i]
        donchian_lower_4h_val = donchian_lower_4h_aligned[i]
        rsi_4h_val = rsi_4h_aligned[i]
        
        # 1h values
        rsi_val = rsi_1h[i]
        price = close[i]
        bbw_val = bbw_1h[i]
        vol = volume[i]
        vol_sma = volume_sma_1h[i]
        
        # 4h trend filter using KAMA
        trend_4h = 0
        if kama_4h_val > 0 and close_4h[min(i // 4, len(close_4h) - 1)] > kama_4h_val:
            trend_4h = 1
        elif kama_4h_val > 0 and close_4h[min(i // 4, len(close_4h) - 1)] < kama_4h_val:
            trend_4h = -1
        
        # 4h RSI filter
        rsi_4h_bullish = rsi_4h_val > RSI_4H_BULL_MIN
        rsi_4h_bearish = rsi_4h_val < RSI_4H_BEAR_MAX
        
        # BBW filter - avoid choppy markets
        if bbw_val < BBW_MIN:
            continue
        
        # Volume filter
        if vol_sma > 0 and vol < vol_sma * VOLUME_MULT:
            continue
        
        # Entry logic: 4h trend + 1h pullback + Donchian breakout
        donchian_mid_1h = (donchian_upper_1h[i] + donchian_lower_1h[i]) / 2
        
        # Bullish entry: 4h uptrend + 1h RSI pullback + price above Donchian mid
        if trend_4h == 1 and rsi_4h_bullish:
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and price > donchian_mid_1h):
                signals[i] = SIZE_FULL
                
        # Bearish entry: 4h downtrend + 1h RSI pullback + price below Donchian mid
        elif trend_4h == -1 and rsi_4h_bearish:
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and price < donchian_mid_1h):
                signals[i] = -SIZE_FULL
    
    # Apply stoploss and take profit logic (second pass)
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    ATR_STOP_MULT = 2.0
    
    for i in range(first_valid, n):
        if signals[i] != 0 and position_side == 0:
            # New entry
            position_side = 1 if signals[i] > 0 else -1
            entry_price = close[i]
            tp_triggered = False
            highest_since_entry = close[i]
            lowest_since_entry = close[i]
        elif position_side != 0:
            # Update highest/lowest since entry
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i]) if lowest_since_entry > 0 else close[i]
            else:
                highest_since_entry = max(highest_since_entry, close[i]) if highest_since_entry > 0 else close[i]
                lowest_since_entry = min(lowest_since_entry, close[i])
            
            # Get current ATR
            atr = atr_1h[i]
            if np.isnan(atr) or atr == 0:
                atr = atr_1h[max(first_valid, i - 1)]
            
            # Stoploss check
            if position_side == 1:
                stoploss_price = entry_price - ATR_STOP_MULT * atr
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = entry_price + 2 * ATR_STOP_MULT * atr
                if not tp_triggered and close[i] >= tp_price:
                    signals[i] = SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if tp_triggered:
                    trail_stop = highest_since_entry - ATR_STOP_MULT * atr
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
                
            elif position_side == -1:
                stoploss_price = entry_price + ATR_STOP_MULT * atr
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = entry_price - 2 * ATR_STOP_MULT * atr
                if not tp_triggered and close[i] <= tp_price:
                    signals[i] = -SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if tp_triggered:
                    trail_stop = lowest_since_entry + ATR_STOP_MULT * atr
                    if close[i] > trail_stop:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
    
    return signals