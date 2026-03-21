#!/usr/bin/env python3
"""
EXPERIMENT #018 - MTF KAMA+Zscore+DailySMA (1h+1d v1)
==================================================================================================
Hypothesis: 1h primary timeframe with KAMA (Kaufman Adaptive Moving Average) for trend +
Daily SMA(50) for institutional trend filter + Z-score(20) for mean reversion entry timing.

Why this should beat current best (mtf_supertrend_rsi_daily_1h_4h_1d_v1, Sharpe=0.065):
- KAMA adapts to volatility (ER-based), better than static Supertrend in choppy markets
- Z-score captures mean reversion opportunities (different from RSI pullback)
- 1h primary generates more trades than 4h while less noisy than 15m/30m
- Daily SMA(50) is proven institutional support/resistance level
- Simpler logic = less overfitting risk

Key differences from failed strategies:
- Not using 30m (too noisy, failed in #006, #011, #012)
- Not using complex multi-indicator combos (ADX+Stoch+Volume failed in #013, #016)
- Using KAMA instead of HMA/DEMA (adaptive vs fixed)
- Using Z-score instead of RSI for entry timing (different signal type)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_zscore_daily_sma_1h_1d_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility using Efficiency Ratio (ER)
    
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(1, er_period + 1):
            noise += abs(close[i - j + 1] - close[i - j])
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
        else:
            zscore[i] = 0.0
    
    return zscore


def calculate_sma(close, period=50):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    zscore_1h = calculate_zscore(close, period=20)
    
    # Get 1d data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # Daily SMA(50) for institutional trend filter
        sma_50_1d = calculate_sma(c_1d, period=50)
        
        # Align daily indicators to 1h timeframe (auto shift for completed bars)
        sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
        c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
        
    except Exception:
        # Fallback if mtf_data fails
        sma_50_1d_aligned = np.zeros(n)
        c_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # Z-score thresholds for mean reversion entries
    ZSCORE_LONG_MAX = -0.5  # Enter when price is below mean (oversold in uptrend)
    ZSCORE_SHORT_MIN = 0.5  # Enter when price is above mean (overbought in downtrend)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 50, 30, 20)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(kama_1h[i]) or np.isnan(zscore_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned daily values
        daily_price = c_1d_aligned[i] if i < len(c_1d_aligned) else 0
        daily_sma = sma_50_1d_aligned[i] if i < len(sma_50_1d_aligned) else 0
        
        # Daily trend filter (price vs SMA50)
        daily_trend = 0
        if daily_price > 0 and daily_sma > 0:
            if daily_price > daily_sma:
                daily_trend = 1  # Bullish
            elif daily_price < daily_sma:
                daily_trend = -1  # Bearish
        
        # KAMA trend direction on 1h
        kama_trend = 0
        if i >= 2 and kama_1h[i] > 0 and kama_1h[i-1] > 0:
            if close[i] > kama_1h[i] and kama_1h[i] > kama_1h[i-1]:
                kama_trend = 1  # Bullish
            elif close[i] < kama_1h[i] and kama_1h[i] < kama_1h[i-1]:
                kama_trend = -1  # Bearish
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered and trend still valid
            if daily_trend == prev_side or daily_trend == 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                # Close position if daily trend reverses
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            continue
        
        # Entry logic: Daily trend + 1h KAMA trend + Z-score mean reversion
        price = close[i]
        
        if daily_trend == 1 and kama_trend == 1:  # Bullish on both timeframes
            # Z-score negative (price below mean = pullback entry)
            if zscore_1h[i] < ZSCORE_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif daily_trend == -1 and kama_trend == -1:  # Bearish on both timeframes
            # Z-score positive (price above mean = pullback entry)
            if zscore_1h[i] > ZSCORE_SHORT_MIN:
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