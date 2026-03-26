#!/usr/bin/env python3
"""
Experiment #002: 12h Donchian Breakout + Volume + 1d Trend

HYPOTHESIS: On 12h timeframe, price closing outside a 20-period Donchian 
channel marks institutional accumulation/distribution. The 12h pace (2 bars/day)
captures meaningful breakouts while avoiding noise that plagues faster TFs.
Volume confirmation filters false breakouts. 1d HMA aligns with swing trades.
The exit on channel re-entry is simple and captures full moves.

Why 12h over 4h: 2 bars/day = ~60 entry opportunities/year vs 6 for 4h.
But Donchian(20) on 12h = equivalent structure to Donchian(20) on 4h in terms of
price action (20 periods regardless of timeframe). 12h reduces data noise.

TIMEFRAME: 12h
HTF: 1d for trend bias
TARGET: 75-200 total trades over 4 years (19-50/year)
SIZE: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_v2"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper = np.full(n, np.nan, dtype=np.float64)
    donch_middle = np.full(n, np.nan, dtype=np.float64)
    donch_lower = np.full(n, np.nan, dtype=np.float64)
    
    period = 20
    for i in range(period - 1, n):
        upper_val = np.max(high[i - period + 1:i + 1])
        lower_val = np.min(low[i - period + 1:i + 1])
        donch_upper[i] = upper_val
        donch_lower[i] = lower_val
        donch_middle[i] = (upper_val + lower_val) / 2.0
    
    # Volume MA for confirmation
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
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.15
        
        # === DONCHIAN CHANNEL ANALYSIS ===
        # Price outside channel?
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # Price inside channel?
        price_inside = not price_above_upper and not price_below_lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Price breaks above upper band + bullish 1d trend + volume
            if price_above_upper and trend_bullish and vol_spike:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Price breaks below lower band + bearish 1d trend + volume
            if price_below_lower and trend_bearish and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        if in_position and desired_signal == 0.0:
            # Check if stoploss is hit before checking exits
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    desired_signal = 0.0  # Stopped out
                elif price_inside:
                    # Exit on channel re-entry
                    desired_signal = 0.0
            
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    desired_signal = 0.0  # Stopped out
                elif price_inside:
                    # Exit on channel re-entry
                    desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
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
        
        signals[i] = desired_signal
    
    return signals