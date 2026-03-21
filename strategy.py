#!/usr/bin/env python3
"""
EXPERIMENT #016 - MTF Donchian+RSI+Volume (1h+4h+1d v1)
==================================================================================================
Hypothesis: Use 1h primary timeframe (more trades than 4h, less noise than 15m) with:
- 4h Donchian channel for trend direction (breakout-based, clearer than HMA)
- 1d BBW for regime filter (avoid low volatility days)
- 1h RSI pullback for entries (proven in #009 which had Sharpe=0.065)
- Volume confirmation to filter fake breakouts

Why this should work:
- 1h timeframe balances trade frequency vs noise (current best #009 uses 1h)
- Donchian channels capture breakouts better than moving averages in crypto
- Daily BBW filter avoids choppy market days (reduces whipsaws)
- Volume confirmation adds conviction to entries
- Simpler logic than #004 = more trades (avoiding the <10 trades failure mode)

Key differences from current best (#009):
- Donchian trend instead of Supertrend
- Volume filter added
- 1h primary (same as #009) but different entry logic
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_rsi_volume_1h_4h_1d_v1"
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)"""
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
    for i in range(n):
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation"""
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
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    donchian_upper_1h, donchian_lower_1h = calculate_donchian(high, low, period=20)
    volume_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Get 4h data for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h Donchian for trend direction
        donchian_upper_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
        
        # Calculate 4h trend (price position in Donchian channel)
        trend_4h = np.zeros(len(c_4h))
        for i in range(len(c_4h)):
            if donchian_upper_4h[i] > 0 and donchian_lower_4h[i] > 0:
                channel_mid = (donchian_upper_4h[i] + donchian_lower_4h[i]) / 2
                if c_4h[i] > channel_mid:
                    trend_4h[i] = 1
                elif c_4h[i] < channel_mid:
                    trend_4h[i] = -1
        
        # Align 4h trend to 1h timeframe
        trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    except Exception:
        trend_4h_aligned = np.zeros(n)
    
    # Get 1d data for regime filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # 1d BBW for regime filter
        _, _, _, bbw_1d = calculate_bollinger_bands(c_1d, period=20, std_mult=2.0)
        
        # Align 1d BBW to 1h timeframe
        bbw_1d_aligned = align_htf_to_ltf(prices, df_1d, bbw_1d)
    except Exception:
        bbw_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.0875
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45
    RSI_LONG_EXIT = 55
    RSI_SHORT_ENTRY = 55
    RSI_SHORT_EXIT = 45
    
    # Volume confirmation threshold
    VOLUME_MULT = 1.2
    
    # BBW minimum for regime filter (avoid choppy markets on daily)
    BBW_MIN_1D = 0.02
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 20, 14 + 1)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Get aligned MTF values
        trend_4h_val = trend_4h_aligned[i] if i < len(trend_4h_aligned) else 0
        bbw_1d_val = bbw_1d_aligned[i] if i < len(bbw_1d_aligned) else 0
        
        # Daily BBW regime filter - avoid choppy days
        if bbw_1d_val < BBW_MIN_1D:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
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
            
            # Check if trend reversed - close position
            if prev_side == 1 and trend_4h_val == -1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            elif prev_side == -1 and trend_4h_val == 1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # Check RSI exit signals
            if prev_side == 1 and rsi_1h[i] > RSI_LONG_EXIT:
                signals[i] = SIZE_QUARTER
                position_side[i] = 1
                entry_price[i] = prev_entry
                tp_triggered[i] = prev_tp
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
                continue
            elif prev_side == -1 and rsi_1h[i] < RSI_SHORT_EXIT:
                signals[i] = -SIZE_QUARTER
                position_side[i] = -1
                entry_price[i] = prev_entry
                tp_triggered[i] = prev_tp
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
                continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 1h RSI pullback + Volume confirmation
        price = close[i]
        vol_ratio = volume[i] / volume_sma_1h[i] if volume_sma_1h[i] > 0 else 1.0
        
        if trend_4h_val == 1:  # Bullish trend on 4h
            # RSI pullback on 1h (not overbought) + Volume confirmation
            if (RSI_LONG_ENTRY <= rsi_1h[i] <= RSI_LONG_EXIT and 
                vol_ratio >= VOLUME_MULT):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h_val == -1:  # Bearish trend on 4h
            # RSI pullback on 1h (not oversold) + Volume confirmation
            if (RSI_SHORT_EXIT <= rsi_1h[i] <= RSI_SHORT_ENTRY and 
                vol_ratio >= VOLUME_MULT):
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