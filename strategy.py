#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Volume + 1d HMA Trend + Choppiness Filter

HYPOTHESIS: 4h timeframe generates more trades than 12h (proven from DB:
95 trades over 4 years = ~24/year is optimal). Donchian(20) breakout with
volume spike (>1.5x) captures institutional moves. 1d HMA(21) as trend filter
ensures trades align with higher timeframe direction. Choppiness Index
(<50 = trending) avoids range-bound whipsaws. ATR-based stoploss at 2.5x.

WHY 4h > 12h: Recent experiments show 12h generates only 29-72 trades (too few
for statistical validity). DB shows 4h strategies with 75-100 trades perform
best on test (Sharpe 1.3-1.47).

TARGET: 75-150 total trades over 4 years (19-38/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_1d_v1"
timeframe = "4h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, middle, lower

def calculate_choppiness(close, high, low, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy (mean reversion mode)
    CHOP < 38.2 = trending (trend following mode)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            idx = i - j
            tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]), abs(low[idx] - close[idx-1]))
            sum_tr += tr
        
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        range_val = highest_high - lowest_low
        
        if range_val > 1e-10:
            chop[i] = 100.0 * (np.log(sum_tr) / np.log(range_val * period)) if range_val > 0 else np.nan
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias (bull/bear filter)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_middle, donch_lower = calculate_donchian(high, low, period=20)
    
    # Choppiness Index - trending filter
    chop = calculate_choppiness(close, high, low, period=14)
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if key indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(hma_1d_aligned[i]):
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
        
        # === TREND FILTER (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        trend_bullish = price_above_1d_hma
        trend_bearish = not price_above_1d_hma
        
        # === REGIME FILTER (Choppiness) ===
        # CHOP < 50 = trending (good for breakouts)
        # CHOP >= 50 = choppy (avoid breakout trades)
        is_trending = chop[i] < 50.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout: price closes above/below 20-bar channel
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # Previous bar position check
        prev_above_upper = close[i-1] > donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else False
        prev_below_lower = close[i-1] < donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else False
        
        # True breakout: cross through channel
        breakout_up = price_above_upper and not prev_above_upper
        breakout_down = price_below_lower and not prev_below_lower
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === NEW LONG ENTRY ===
            # Breakout above upper channel + volume spike + trending + bullish 1d
            if breakout_up and vol_spike and is_trending and trend_bullish:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Breakout below lower channel + volume spike + trending + bearish 1d
            if breakout_down and vol_spike and is_trending and trend_bearish:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === EXIT: Channel reversion OR trend flip ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price falls below lower channel OR 1d trend turns bearish
            if price_below_lower:
                exit_triggered = True
            if trend_bearish and close[i] < donch_middle[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price rises above upper channel OR 1d trend turns bullish
            if price_above_upper:
                exit_triggered = True
            if trend_bullish and close[i] > donch_middle[i]:
                exit_triggered = True
        
        if exit_triggered:
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