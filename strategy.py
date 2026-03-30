#!/usr/bin/env python3
"""
Experiment #026: 4h Donchian Breakout + Volume + Choppiness Regime

HYPOTHESIS: Donchian(20) breakout is a proven price structure indicator that captures
momentum when price makes a 20-period high/low. Combined with volume confirmation
(volume > 1.5x 20-avg) to filter false breakouts, and Choppiness Index to avoid
trading in range-bound markets, this should work in both bull (breakout continuation)
and bear (breakdown continuation).

WHY 4h: Balances signal quality with trade frequency. 4h has proven best keep rate (41%)
in the DB. 6h/12h often have too few trades.

TARGET: 100-200 total trades over 4 years = 25-50/year. HARD MAX: 400.

KEY INSIGHT FROM DB: Best performers use ONE strong signal (price channel) + volume +
regime filter. Don't overcomplicate.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_simple_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range market (avoid trend following)
    CHOP < 38.2 = trending market (trend following works)
    """
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma50_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Need 20 for Donchian + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma50_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma50_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Trend following works when CHOP < 38.2 (trending)
        # In choppy market, trend following fails
        is_trending = chop[i] < 38.2
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout: close above 20-bar high (bull) or below 20-bar low (bear)
        bull_breakout = close[i] > donchian_high[i-1] if i > 0 else False
        bear_breakout = close[i] < donchian_low[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Price broke above Donchian high + trending + volume confirm
            if bull_breakout and is_trending and price_above_1d_sma and vol_spike:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Price broke below Donchian low + trending + volume confirm
            if bear_breakout and is_trending and not price_above_1d_sma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Long stoploss: price fell below entry - 2*ATR
            stop_price = entry_price - 2.0 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Short stoploss: price rose above entry + 2*ATR
            stop_price = entry_price + 2.0 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TRAILING STOP (when in profit) ===
        if in_position and position_side > 0:
            # Track highest high since entry
            if high[i] > entry_price + 1.5 * entry_atr:
                # In profit, trail stop at 1.5*ATR below highest high
                trailing_stop = high[i] - 2.0 * entry_atr
                if low[i] < trailing_stop:
                    desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Track lowest low since entry
            if low[i] < entry_price - 1.5 * entry_atr:
                # In profit, trail stop at 1.5*ATR above lowest low
                trailing_stop = low[i] + 2.0 * entry_atr
                if high[i] > trailing_stop:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD (4 bars = 16 hours) ===
        bars_held = i - entry_bar
        min_hold = 4
        
        if in_position and bars_held < min_hold:
            # Don't exit early, maintain position
            if desired_signal == 0.0 and not stoploss_triggered:
                desired_signal = position_side * SIZE
        
        # === ATR TRAILING EXIT ===
        if in_position:
            # Exit if ATR-based profit target hit (3:1)
            if position_side > 0:
                profit_target = entry_price + 3.0 * entry_atr
                if high[i] >= profit_target:
                    desired_signal = 0.0
            
            if position_side < 0:
                profit_target = entry_price - 3.0 * entry_atr
                if low[i] <= profit_target:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
            else:
                # Maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
        
        signals[i] = desired_signal
    
    return signals