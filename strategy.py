#!/usr/bin/env python3
"""
Experiment #028: 6h Fisher Transform + Volume Spike + 1d KAMA Trend + Choppiness Regime

HYPOTHESIS: Fisher Transform identifies extreme overbought/oversold conditions in a 
normalized scale. Combined with 1d KAMA trend and volume confirmation, it catches 
high-probability reversal points. Choppiness keeps us out of trending markets.

WHY IT WORKS IN BULL AND BEAR: Bull = buy Fisher oversold during corrections.
Bear = short Fisher overbought during rallies. KAMA trend filter keeps us aligned
with macro direction. Volume confirms institutional participation.

TARGET: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 300.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_kama_vol_chop_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Fisher Transform - normalized oscillator that identifies price extremes.
    Values typically range from -3 to +3.
    Long when Fisher crosses above -1.5 (signal line), short when below +1.5.
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n, dtype=np.float64)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        highest = -np.inf
        lowest = np.inf
        
        for j in range(i - period + 1, i + 1):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]
        
        if highest > lowest:
            hl2 = (high[i] + low[i]) / 2
            x = 0.33 * 2 * ((hl2 - lowest) / (highest - lowest) - 0.5) + 0.67 * signal[i - 1]
            x = np.clip(x, -0.999, 0.999)
            fisher[i] = 0.5 * np.log((1 + x) / (1 - x)) if x != 0 else 0
            signal[i] = fisher[i - 1]  # Trigger line is prior Fisher value
    
    return fisher, signal

def calculate_kama(close, period=30, fast_ema=2, slow_ema=30):
    """
    Kaufman's Adaptive Moving Average - adapts to market volatility.
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period, dtype=np.float64)
    for i in range(n - period):
        for j in range(period):
            volatility[i] += np.abs(close[i + j + 1] - close[i + j])
    
    er = np.zeros(n, dtype=np.float64)
    er[period:] = direction / np.where(volatility > 0, volatility, 1)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_const = 2 / (fast_ema + 1)
    slow_const = 2 / (slow_ema + 1)
    smoothing = er * (fast_const - slow_const) + slow_const
    smoothing_sq = smoothing ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + smoothing_sq[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend direction (adaptive = better than SMA)
    kama_1d = calculate_kama(df_1d['close'].values, period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Local 6h indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d KAMA) ===
        price_above_1d_kama = close[i] > kama_1d_aligned[i]
        bull_trend = price_above_1d_kama
        bear_trend = not price_above_1d_kama
        
        # === REGIME (Choppiness Index) ===
        # Only trade when not too choppy
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 50.0
        
        # Skip if too choppy (except for exits)
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === FISHER TRANSFORM SIGNALS ===
        current_fisher = fisher[i]
        prev_fisher = fisher[i - 1] if i > 0 else 0
        current_signal = fisher_signal[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Fisher crosses above -1.5 (bullish reversal)
            # Only in bull trend or when coming from oversold
            if prev_fisher <= current_signal and current_fisher > current_signal:
                if current_fisher < -1.0:  # Must be in oversold territory
                    if vol_spike or is_trending:  # Volume or momentum confirmation
                        if bull_trend:  # Align with 1d trend
                            desired_signal = SIZE
                        elif current_fisher < -1.5:  # Strong oversold can override trend
                            desired_signal = SIZE * 0.5  # Half size against trend
            
            # === SHORT: Fisher crosses below +1.5 (bearish reversal)
            # Only in bear trend or when coming from overbought
            if prev_fisher >= current_signal and current_fisher < current_signal:
                if current_fisher > 1.0:  # Must be in overbought territory
                    if vol_spike or is_trending:  # Volume or momentum confirmation
                        if bear_trend:  # Align with 1d trend
                            desired_signal = -SIZE
                        elif current_fisher > 1.5:  # Strong overbought can override trend
                            desired_signal = -SIZE * 0.5  # Half size against trend
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit if Fisher reverts to mean (crosses 0)
            if position_side > 0 and current_fisher > 0:
                desired_signal = 0.0
            if position_side < 0 and current_fisher < 0:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals