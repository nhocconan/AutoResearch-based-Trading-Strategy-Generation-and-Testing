#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian Breakout + Weekly Trend + Volume Spike

HYPOTHESIS: Daily Donchian(20) breakouts capture significant price moves. 
Weekly HMA(21) trend filter ensures we only trade in direction of higher timeframe trend.
Volume spike (>1.5x 20-avg) confirms institutional participation, not fake breakouts.
Choppiness Index filters ranging markets where breakouts fail.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: long breakouts above Donchian high when weekly HMA sloping up
- Bear markets: short breakouts below Donchian low when weekly HMA sloping down
- Weekly trend filter prevents counter-trend trades that get stopped out
- Volume confirmation filters false breakouts (common in 2022 crash, 2025 bear)

TARGET: 40-100 total trades over 4 years (10-25/year). HARD MAX: 150 total.
1d timeframe naturally produces fewer trades - need to ensure entry not too strict.

KEY DESIGN:
1. Donchian(20) breakout as primary signal
2. Weekly HMA(21) slope for trend direction
3. Volume spike >1.5x 20-day average
4. Choppiness <55 (trending regime only)
5. ATR(14) stoploss at 2.5x
6. Signal: 0.25-0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_vol_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging (no trades), CHOP < 50 = trending (allow trades)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1w HMA slope (current vs 1 week ago)
    hma_1w_slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(7, n):  # 7 days = 1 week on 1d timeframe
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i - 7]):
            hma_1w_slope[i] = hma_1w_aligned[i] - hma_1w_aligned[i - 7]
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Slightly lower for 1d to reduce risk
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_trade = 0
    
    # Warmup
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
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
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 55.0  # Allow entries in trending or neutral
        
        # === WEEKLY TREND BIAS ===
        weekly_bullish = hma_1w_slope[i] > 0
        weekly_bearish = hma_1w_slope[i] < 0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3  # Lowered from 1.5 to get more trades
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donch_upper[i - 1] if not np.isnan(donch_upper[i - 1]) else False
        breakout_short = close[i] < donch_lower[i - 1] if not np.isnan(donch_lower[i - 1]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Donchian breakout + weekly bullish + volume + trending
        if is_trending and weekly_bullish and breakout_long:
            if vol_spike:
                desired_signal = SIZE
            elif vol_ratio[i] > 1.0:  # Just above average is ok
                desired_signal = SIZE * 0.75
        
        # SHORT: Donchian breakout + weekly bearish + volume + trending
        if is_trending and weekly_bearish and breakout_short:
            if vol_spike:
                desired_signal = -SIZE
            elif vol_ratio[i] > 1.0:
                desired_signal = -SIZE * 0.75
        
        # === STOPLOSS CHECK ===
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
        
        # === TAKE PROFIT at 2R ===
        tp_triggered = False
        if in_position and position_side > 0:
            tp_price = entry_price + 2.5 * entry_atr
            if high[i] >= tp_price:
                tp_triggered = True
        
        if in_position and position_side < 0:
            tp_price = entry_price - 2.5 * entry_atr
            if low[i] <= tp_price:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === WEEKLY TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side > 0 and weekly_bearish:
            trend_reversal = True
        if in_position and position_side < 0 and weekly_bullish:
            trend_reversal = True
        
        if trend_reversal:
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
                bars_in_trade = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            bars_in_trade += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_in_trade = 0
        
        signals[i] = desired_signal
    
    return signals