#!/usr/bin/env python3
"""
EXPERIMENT #039 - MTF DONCHIAN+KAMA+RSI 4h+1d v1
==================================================================================================
Hypothesis: 4h primary timeframe with 1d trend filter should produce cleaner signals with fewer 
whipsaws than 1h strategies. This approach:
- Uses 4h base (fewer trades than 1h/30m, but higher quality signals)
- 1d Donchian(20) for major trend direction (daily breakouts are significant)
- 4h KAMA(21) for adaptive trend following (adjusts to volatility)
- 4h RSI(14) for pullback entries with wider thresholds suitable for 4h
- Position size: 0.35 (conservative, discrete levels)

Why this should beat current best:
- 1d trend filter is stronger than 4h (fewer false trend reversals)
- Donchian channels capture breakout momentum better than HMA alone
- KAMA adapts to volatility changes (better in choppy vs trending markets)
- 4h timeframe = ~6x fewer signals than 1h = lower transaction costs
- Proven combo: Donchian trends + RSI pullbacks worked well in #027
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_kama_rsi_4h_1d_v1"
timeframe = "4h"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                noise += abs(close[j] - close[j - 1])
        
        if noise > 0:
            signal = abs(close[i] - close[i - period])
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
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
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 4h indicators for entry timing
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    kama_4h = calculate_kama(close, period=21, fast=2, slow=30)
    donchian_upper_4h, donchian_lower_4h = calculate_donchian_channels(high, low, period=20)
    
    # Get 1d data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        
        # 1d Donchian for major trend direction
        donchian_upper_1d, donchian_lower_1d = calculate_donchian_channels(h_1d, l_1d, period=20)
        donchian_mid_1d = (donchian_upper_1d + donchian_lower_1d) / 2
        
        # Align 1d indicators to 4h timeframe (auto shift for completed bars)
        donchian_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_1d)
        donchian_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
        donchian_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
        
    except Exception:
        # Fallback if mtf_data fails
        donchian_mid_1d_aligned = np.zeros(n)
        donchian_upper_1d_aligned = np.zeros(n)
        donchian_lower_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries (wider for 4h timeframe)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 50
    RSI_SHORT_MIN = 50
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5  # Wider stop for 4h timeframe
    
    first_valid = max(100, 30 * 2, 20 * 2, 14 * 3)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_4h[i]) or np.isnan(rsi_4h[i]) or atr_4h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values
        donchian_mid_1d_val = donchian_mid_1d_aligned[i] if i < len(donchian_mid_1d_aligned) else 0
        donchian_upper_1d_val = donchian_upper_1d_aligned[i] if i < len(donchian_upper_1d_aligned) else 0
        donchian_lower_1d_val = donchian_lower_1d_aligned[i] if i < len(donchian_lower_1d_aligned) else 0
        
        # 1d trend direction (price vs Donchian mid)
        trend_1d = 0
        if close[i] > donchian_mid_1d_val and donchian_mid_1d_val > 0:
            trend_1d = 1
        elif close[i] < donchian_mid_1d_val and donchian_mid_1d_val > 0:
            trend_1d = -1
        
        # 4h KAMA trend confirmation
        kama_trend = 0
        if kama_4h[i] > 0:
            if close[i] > kama_4h[i]:
                kama_trend = 1
            elif close[i] < kama_4h[i]:
                kama_trend = -1
        
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
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_4h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_4h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_4h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_4h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_4h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_4h[i]
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
        
        # Entry logic: 1d trend + 4h KAMA confirmation + 4h RSI pullback
        price = close[i]
        
        # Both 1d and 4h must agree on trend direction
        if trend_1d == 1 and kama_trend == 1:  # Bullish trend
            # RSI pullback on 4h (not overbought)
            if RSI_LONG_MIN <= rsi_4h[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_1d == -1 and kama_trend == -1:  # Bearish trend
            # RSI pullback on 4h (not oversold)
            if RSI_SHORT_MIN <= rsi_4h[i] <= RSI_SHORT_MAX:
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