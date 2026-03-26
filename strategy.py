#!/usr/bin/env python3
"""
Experiment #025: 4h Camarilla Bounce + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels (S3/S4/R3/R4) represent mathematically derived
support/resistance based on the prior day's range. Price bounces from these levels
exhibit mean-reversion behavior - when price touches S3/S4 with RSI oversold,
it tends to revert back toward the daily open. Combined with:
- Choppiness filter (avoid trades in choppy markets)
- Volume confirmation (validates institutional interest)
- ATR stoploss (tight risk management)

This is DIFFERENT from failed Donchian breakouts because:
- Entry ON the level (bounce), not BREAKOUT through the level
- Mean-reversion logic, not trend-following
- Fewer signals = less fee drag

TIMEFRAME: 4h primary
HTF: 1d for trend context
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_bounce_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla_pivots(open_price, high, low, close):
    """
    Calculate Camarilla pivot levels.
    These are based on the prior day's range and open.
    S3 = Low + (High - Open) * 1.1/3
    S4 = Low + (High - Open) * 1.1/6
    R3 = High - (High - Open) * 1.1/3
    R4 = High - (High - Open) * 1.1/6
    """
    n = len(open_price)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    
    for i in range(1, n):
        if np.isnan(open_price[i-1]) or np.isnan(high[i-1]) or np.isnan(low[i-1]):
            continue
        day_range = high[i-1] - low[i-1]
        day_open = open_price[i-1]
        day_low = low[i-1]
        day_high = high[i-1]
        
        # Camarilla formula
        s3[i] = day_low + day_range * 1.1 / 3
        s4[i] = day_low + day_range * 1.1 / 6
        r3[i] = day_high - day_range * 1.1 / 3
        r4[i] = day_high - day_range * 1.1 / 6
    
    return s3, s4, r3, r4

def calculate_choppiness(close, high, low, period=14):
    """
    Choppiness Index - measures market "choppiness" vs trending.
    < 38.2 = trending market
    > 61.8 = choppy/ranging market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of true ranges
        sum_tr = 0.0
        for j in range(i - period + 1, i + 1):
            tr = high[j] - low[j]
            if j > 0:
                tr = max(tr, abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            sum_tr += tr
        
        # Highest high - lowest low over period
        highest = max(high[i - period + 1:i + 1])
        lowest = min(low[i - period + 1:i + 1])
        range_sum = highest - lowest
        
        if range_sum > 0:
            chop[i] = 100 * np.log10(sum_tr / range_sum) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """RSI indicator"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_prices = prices["open"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA for trend bias (price above = bullish)
    sma_1d_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # === Calculate local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(close, high, low, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Camarilla pivots (using prior 4h bar's open/high/low)
    s3, s4, r3, r4 = calculate_camarilla_pivots(open_prices, high, low, close)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    partial_exit_done = False
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s3[i]) or np.isnan(r3[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === Regime check: choppiness < 61.8 (not too choppy) ===
        chop_trending = chop[i] < 61.8
        
        # === Volume confirmation ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === RSI values ===
        rsi_val = rsi[i]
        
        # === Trend bias from 1d SMA ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i] if not np.isnan(sma_1d_aligned[i]) else True
        
        # === Camarilla level touch detection ===
        # Price within 0.5 ATR of the level = "touch"
        touch_tolerance = 0.5 * atr_14[i]
        
        # Long entry: price touches S3 or S4
        touch_s3_long = (close[i] >= s3[i] - touch_tolerance) and (close[i] <= s3[i] + touch_tolerance)
        touch_s4_long = (close[i] >= s4[i] - touch_tolerance) and (close[i] <= s4[i] + touch_tolerance)
        near_support = touch_s3_long or touch_s4_long
        
        # Short entry: price touches R3 or R4
        touch_r3_short = (close[i] >= r3[i] - touch_tolerance) and (close[i] <= r3[i] + touch_tolerance)
        touch_r4_short = (close[i] >= r4[i] - touch_tolerance) and (close[i] <= r4[i] + touch_tolerance)
        near_resistance = touch_r3_short or touch_r4_short
        
        # === RSI confirmation at extremes ===
        rsi_oversold = rsi_val < 40
        rsi_overbought = rsi_val > 60
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Price touches S3/S4 + RSI oversold + volume spike + not too choppy
            if near_support and rsi_oversold and vol_spike and chop_trending:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Price touches R3/R4 + RSI overbought + volume spike + not too choppy
            if near_resistance and rsi_overbought and vol_spike and chop_trending:
                desired_signal = -SIZE
        
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
        
        # === TAKE PROFIT / EXIT ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Exit long: RSI reaches overbought OR price hits R3/R4
            if rsi_val > 65:
                exit_triggered = True
            # Or price returns to R3/R4 zone (mean reversion complete)
            if near_resistance:
                exit_triggered = True
            # Partial exit at 2R profit
            if not partial_exit_done and close[i] >= entry_price + 2.0 * entry_atr:
                desired_signal = SIZE / 2  # Half position
                partial_exit_done = True
        
        if in_position and position_side < 0:
            # Exit short: RSI reaches oversold OR price hits S3/S4
            if rsi_val < 35:
                exit_triggered = True
            # Or price returns to S3/S4 zone (mean reversion complete)
            if near_support:
                exit_triggered = True
            # Partial exit at 2R profit
            if not partial_exit_done and close[i] <= entry_price - 2.0 * entry_atr:
                desired_signal = -SIZE / 2  # Half position
                partial_exit_done = True
        
        if exit_triggered:
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
                partial_exit_done = False
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                partial_exit_done = False
        
        signals[i] = desired_signal
    
    return signals