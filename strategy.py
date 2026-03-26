#!/usr/bin/env python3
"""
Experiment #011: 6h Donchian Breakout + Daily ATR Volatility Filter + Volume

HYPOTHESIS: Donchian breakouts capture momentum shifts when price breaks the 24-bar
(6h) channel. The daily ATR volatility filter ensures we only trade when volatility
is sufficient (ATR above 30th percentile of 20d) — avoiding false breakouts in
compressed markets. Volume confirms institutional involvement.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: Long breakouts above 1d EMA(50) when price pulls back to upper band
- Bear markets: Short breakouts below 1d EMA(50) when price rallies to lower band
- Volatility filter avoids 2022 crash whipsaws (low vol before big moves)
- 6h timeframe naturally filters noise vs 4h

TARGET: 50-150 total trades over 4 years (12-37/year on 6h).
6h bars/year = 365*4/0.25 = 5,840 bars
Entry rate: ~1-3% of bars = 58-175 entries → with filters → 50-150 trades

KEY DESIGN (simpler = better):
1. 6h Donchian(24) = 4-day channel (aligns with daily structure)
2. Daily ATR volatility regime filter (avoid low-vol chop)
3. Volume spike confirmation (>1.5x 20-avg)
4. Daily EMA(50) trend filter
5. ATR-based stoploss (3x ATR)
6. Signal: 0.25 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_atr_vol_1d_v1"
timeframe = "6h"
leverage = 1.0

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

def calculate_donchian(high, low, period):
    """Donchian Channel"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for ATR filter and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR for volatility regime
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d_pct = pd.Series(atr_1d / df_1d['close'].values * 100).rolling(window=20, min_periods=20).apply(
        lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x) + 1e-10) * 100, raw=True
    ).values
    
    # Align to 6h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_pct)
    
    # 1d EMA(50) for trend
    ema_1d_50 = calculate_ema(df_1d['close'].values, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian(24) on 6h = 4-day channel
    donch_upper, donch_lower = calculate_donchian(high, low, period=24)
    
    # Mid channel
    donch_mid = (donch_upper + donch_lower) / 2
    
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
    
    # Warmup for 1d alignment (24*4 = 96 bars + EMA(50) + vol MA)
    warmup = 120
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY REGIME FILTER ===
        # Only trade when daily ATR is above 30th percentile (avoid chop)
        vol_regime_ok = atr_1d_aligned[i] > 30.0
        
        # === TREND FILTER ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i] if not np.isnan(ema_1d_aligned[i]) else True
        price_below_1d_ema = close[i] < ema_1d_aligned[i] if not np.isnan(ema_1d_aligned[i]) else False
        
        # === DONCHIAN LEVELS ===
        upper = donch_upper[i]
        lower = donch_lower[i]
        mid = donch_mid[i]
        bandwidth = upper - lower
        
        # Price position in channel (0 = bottom, 1 = top)
        if bandwidth > 1e-10:
            price_position = (close[i] - lower) / bandwidth
        else:
            price_position = 0.5
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Price pulls back to lower band in uptrend (bull market)
        # Entry when price is near lower 30% of channel + trend aligns
        if vol_regime_ok:
            # Pullback long: price in lower 30% of channel
            if price_position < 0.30 and price_above_1d_ema:
                if vol_spike:
                    desired_signal = SIZE
                elif i > 0 and close[i] > close[i-1]:  # also confirm price stabilizing
                    desired_signal = SIZE
            
            # Breakout continuation: price was below mid, now above mid
            if mid is not None and not np.isnan(mid):
                if i > 0:
                    prev_pos = (close[i-1] - lower) / bandwidth if bandwidth > 1e-10 else 0.5
                    if prev_pos < 0.5 and price_position >= 0.5 and price_above_1d_ema:
                        if vol_spike:
                            desired_signal = SIZE
        
        # SHORT: Price rallies to upper band in downtrend (bear market)
        if vol_regime_ok:
            # Rally short: price in upper 30% of channel
            if price_position > 0.70 and price_below_1d_ema:
                if vol_spike:
                    desired_signal = -SIZE
                elif i > 0 and close[i] < close[i-1]:  # price stabilizing down
                    desired_signal = -SIZE
            
            # Breakout continuation: price was above mid, now below mid
            if mid is not None and not np.isnan(mid):
                if i > 0:
                    prev_pos = (close[i-1] - lower) / bandwidth if bandwidth > 1e-10 else 0.5
                    if prev_pos > 0.5 and price_position <= 0.5 and price_below_1d_ema:
                        if vol_spike:
                            desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT at opposite channel ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at upper band or beyond mid significantly
            if high[i] >= upper:
                tp_triggered = True
            elif price_position > 0.85:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at lower band or beyond mid significantly
            if low[i] <= lower:
                tp_triggered = True
            elif price_position < 0.15:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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