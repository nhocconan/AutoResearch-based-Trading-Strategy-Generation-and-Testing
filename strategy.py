#!/usr/bin/env python3
"""
Experiment #010: 1d Primary + 1w HTF — Simple Donchian Breakout with Weekly Trend

HYPOTHESIS:
1d timeframe with 1w trend bias captures the best risk-adjusted trades because:
- 1d = enough bars (~1460/year) for statistical validity without overtrading
- 1w HMA(21) provides strong trend filter without being too slow
- Donchian(20) breakout is a proven edge (SOLUSDT test Sharpe 1.10-1.38)
- Choppiness regime filter prevents trading in ranges (biggest cause of failure)
- Simple = fewer conditions = fewer trades = less fee drag

Why this should work in BOTH bull AND bear:
- Bull: Price breaks Donchian high + 1w trend up = strong momentum trade
- Bear: Price breaks Donchian low + 1w trend down = continuation short
- Range: CHOP>61 prevents entries in chop, avoids 2022 whipsaw destruction
- ATR stoploss (2.5x) handles volatility expansion during crashes

Target trades: 30-100 total over 4 years (7-25/year) - HARD MAX: 150
Entry philosophy: ONLY enter on confirmed Donchian breakout, not on pullbacks
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_trend_chop_v1"
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
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging (don't trade), CHOP < 38.2 = trending (trade)
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
    """Donchian Channel - price channel breakout system"""
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
    """
    Volume ratio - compares current volume to average
    Used to confirm breakouts (high volume = more likely valid)
    """
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    avg_vol = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(avg_vol[i]) and avg_vol[i] > 0 and volume[i] > 0:
            ratio[i] = volume[i] / avg_vol[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position tracking for stoploss and trailing stop
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period - need enough bars for all indicators
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(hma_1w_aligned[i]):
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
        
        # === 1w TREND DIRECTION ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME CHECK (only trade in trending regime) ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Relaxed threshold - only filter extreme chop
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Need previous bar's Donchian for breakout confirmation
        donch_upper_prev = donch_upper[i-1] if i > 0 and not np.isnan(donch_upper[i-1]) else np.nan
        donch_lower_prev = donch_lower[i-1] if i > 0 and not np.isnan(donch_lower[i-1]) else np.nan
        
        # Breakout: price CLOSES above/below previous period's channel
        donch_breakout_long = False
        donch_breakout_short = False
        
        if not np.isnan(donch_upper_prev):
            donch_breakout_long = close[i] > donch_upper_prev
        if not np.isnan(donch_lower_prev):
            donch_breakout_short = close[i] < donch_lower_prev
        
        # === VOLUME CONFIRMATION ===
        # Volume ratio > 1.2 = above average volume (confirms breakout)
        vol_confirmed = vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        # Simple, strict: ONLY enter on confirmed Donchian breakout + 1w trend
        # This guarantees we don't overtrade
        desired_signal = 0.0
        
        # LONG: 1w trend up + Donchian breakout + volume confirmation
        if price_above_1w and donch_breakout_long and vol_confirmed and is_trending:
            desired_signal = 0.30
        
        # SHORT: 1w trend down + Donchian breakout + volume confirmation
        elif price_below_1w and donch_breakout_short and vol_confirmed and is_trending:
            desired_signal = -0.30
        
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
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= 0.25:
            final_signal = 0.30
        elif desired_signal <= -0.25:
            final_signal = -0.30
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals