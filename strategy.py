#!/usr/bin/env python3
"""
Experiment #002: 12h Primary + 1d HTF — Donchian Breakout with Trend Confirmation

HYPOTHESIS: 12h timeframe with strict 1d trend filter will generate 50-100 trades/4yr.
Donchian(24) breakout catches structural breaks. 1d HMA ensures we only trade with 
the larger trend. Choppiness filter eliminates range-bound whipsaws.

Why this should work in BOTH bull AND bear:
- Long entries only when 1d trend is bullish (avoids 2022 crash whipsaws)
- Short entries only when 1d trend is bearish (catches 2022 crash)
- 12h bars = meaningful moves, not noise
- ATR stoploss protects against flash crashes

Key design choices to avoid overtrading:
1. STRICT 1d trend confirmation required (both long AND short)
2. Choppiness < 38.2 required for entries (no choppy market trades)
3. Donchian breakout required (filter out false breakouts)
4. 24-bar Donchian (not 20) = fewer breakouts = fewer trades
5. Hold minimum 2 bars before exit signal

Target: Sharpe > 0.6, trades 50-100 train, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_chop_1d_strict_v1"
timeframe = "12h"
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
    """Choppiness Index - measure market choppiness"""
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

def calculate_donchian(high, low, period=24):
    """Donchian Channel - 24 bars to reduce false breakouts"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs average - identifies spikes"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if vol_avg[i] > 0 and not np.isnan(vol_avg[i]):
            ratio[i] = volume[i] / vol_avg[i]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA and align to 12h
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Also get 1w HMA for additional confirmation
    df_1w = get_htf_data(prices, '1w')
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=10)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=24)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_trade = 0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(donch_upper[i]):
            signals[i] = 0.0
            in_position = False
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            continue
        
        # === TREND DIRECTION (1d HMA + 1w HMA confirmation) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w confirmation (if available)
        hma_1w_ok = not np.isnan(hma_1w_aligned[i])
        price_above_1w = hma_1w_ok and close[i] > hma_1w_aligned[i]
        price_below_1w = hma_1w_ok and close[i] < hma_1w_aligned[i]
        
        # 1d bullish: price above 1d HMA
        trend_bullish = price_above_1d and (not hma_1w_ok or price_above_1w)
        # 1d bearish: price below 1d HMA
        trend_bearish = price_below_1d and (not hma_1w_ok or price_below_1w)
        
        # === CHOPPINESS REGIME ===
        chop = chop_14[i]
        is_trending = chop < 38.2  # Only trade in trending markets
        
        # === DONCHIAN BREAKOUT ===
        donch_prev_upper = donch_upper[i-1] if i > 0 and not np.isnan(donch_upper[i-1]) else np.nan
        donch_prev_lower = donch_lower[i-1] if i > 0 and not np.isnan(donch_lower[i-1]) else np.nan
        
        donch_breakout_long = False
        donch_breakout_short = False
        
        if not np.isnan(donch_prev_upper) and not np.isnan(donch_prev_lower):
            # Breakout: close above prior 24-bar high
            donch_breakout_long = close[i] > donch_prev_upper
            donch_breakout_short = close[i] < donch_prev_lower
        
        # === VOLUME CONFIRMATION (optional, looser) ===
        vol_ok = not np.isnan(vol_ratio[i]) and vol_ratio[i] >= 1.0  # At least average volume
        
        # === ENTRY CONDITIONS (STRICT - minimize trades) ===
        desired_signal = 0.0
        
        # Only enter if: trending market + trend confirmation + breakout
        if is_trending:
            # LONG: Trending + 1d bullish + Donchian breakout
            if trend_bullish and donch_breakout_long:
                desired_signal = SIZE
            
            # SHORT: Trending + 1d bearish + Donchian breakdown
            elif trend_bearish and donch_breakout_short:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # Take profit at 3R if in trade for > 5 bars
        tp_triggered = False
        if in_position and bars_in_trade >= 5:
            if position_side > 0 and (close[i] - entry_price) >= 3.0 * entry_atr:
                tp_triggered = True
            if position_side < 0 and (entry_price - close[i]) >= 3.0 * entry_atr:
                tp_triggered = True
        
        if stoploss_triggered or tp_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE if desired_signal > 0 else -SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_in_trade = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                bars_in_trade += 1
        else:
            if in_position:
                bars_in_trade += 1
            in_position = False
            position_side = 0
        
        signals[i] = final_signal
    
    return signals