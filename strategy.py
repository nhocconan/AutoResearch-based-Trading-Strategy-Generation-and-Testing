#!/usr/bin/env python3
"""
Experiment #011: 6h Donchian Breakout + 1d Trend + Volume Confirmation

HYPOTHESIS: Price breaking OUTSIDE a 6h Donchian channel (not just touching) 
marks institutional accumulation/distribution. Combined with 1d trend direction 
and volume confirmation, this catches major moves while filtering chop.

WHY THIS WORKS IN BOTH BULL AND BEAR:
- Bull: Long breakouts above 6h Donchian high + 1d HMA rising + volume spike
- Bear: Short breakdowns below 6h Donchian low + 1d HMA falling + volume spike
- The key is requiring CLOSE outside channel (not just touching), reducing false signals

TARGET: 75-150 total trades over 4 years (~19-37/year)
HARD MAX: 300 total trades per symbol

KEY INSIGHT FROM FAILURES:
- Entry conditions must be TIGHT (close OUTSIDE channel, not just near it)
- Volume spike required to confirm institutional participation
- 1d trend filter eliminates countertrend trades
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_breakout_1d_v1"
timeframe = "6h"
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
    """Donchian Channel - 20 period (price structure)"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).max().values  # Note: using max of lows for lower band
    
    # Actually lower should be min of lows
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def calculate_hma_slope(hma, period=10):
    """Calculate slope of HMA over last N periods"""
    n = len(hma)
    slope = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if np.isnan(hma[i]) or np.isnan(hma[i - period]):
            continue
        if hma[i - period] > 1e-10:  # Avoid division by zero
            slope[i] = (hma[i] - hma[i - period]) / hma[i - period] * 100
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA(21) for trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1d HMA slope to measure trend strength
    hma_1d_slope_raw = calculate_hma_slope(hma_1d_raw, period=5)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Volume MA (20 period)
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
    bars_in_trade = 0
    
    warmup = 100  # Need 20 bars for Donchian + warmup
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
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
        
        # === INDICATOR VALUES ===
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        dc_range = dc_high - dc_low
        hma_1d = hma_1d_aligned[i]
        hma_slope = hma_1d_slope_aligned[i] if not np.isnan(hma_1d_slope_aligned[i]) else 0.0
        
        # 1d trend: bullish if price above HMA and HMA is rising (positive slope)
        # bearish if price below HMA and HMA is falling (negative slope)
        price_above_1d_hma = close[i] > hma_1d
        price_below_1d_hma = close[i] < hma_1d
        hma_rising = hma_slope > 0.0  # HMA is trending up
        hma_falling = hma_slope < 0.0  # HMA is trending down
        bullish_1d = price_above_1d_hma and hma_rising
        bearish_1d = price_below_1d_hma and hma_falling
        
        # Volume confirmation: spike at least 1.5x average
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT CONDITIONS ===
        # KEY: Close must be OUTSIDE the channel (not just touching)
        # This is much tighter than just being "near" a level
        breakout_above = close[i] > dc_high  # Close above channel
        breakout_below = close[i] < dc_low   # Close below channel
        
        # ATR-based breakout threshold: price should be at least 0.3 ATR beyond channel
        # This prevents entering on marginal breaks
        min_breakout_atr = 0.3
        clear_breakout_above = close[i] > dc_high + min_breakout_atr * atr_14[i]
        clear_breakout_below = close[i] < dc_low - min_breakout_atr * atr_14[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Clear breakout above 6h Donchian + 1d bullish + volume spike
        if clear_breakout_above and bullish_1d and vol_spike:
            desired_signal = SIZE
        
        # SHORT: Clear breakdown below 6h Donchian + 1d bearish + volume spike
        if clear_breakout_below and bearish_1d and vol_spike:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (3 ATR - wider for 6h) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            # Trailing stop: 3 ATR from highest point
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
                bars_in_trade = 0
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
            else:
                # Same side - stay in position (no churn)
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