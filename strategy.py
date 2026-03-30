#!/usr/bin/env python3
"""
Experiment #028: 4h Bollinger Band Squeeze Breakout + Volume + ATR Volatility Filter

HYPOTHESIS: Volatility compression (low Bollinger Bandwidth) precedes major breakouts.
The Squeeze indicator is a classic mean-reversion pattern that identifies low-volatility
condensations before explosive moves. By requiring:
1. BB Width in lower 20th percentile (true compression)
2. Volume expansion on breakout (institutional confirmation)
3. ATR ratio > 1.0 (distinguishes trending from range)
4. 1d SMA200 for trend direction

This strategy captures high-probability breakouts while filtering range noise.

WHY IT WORKS IN BULL AND BEAR:
- Squeeze breakouts fire in BOTH directions - uptrends AND downtrends
- ATR ratio filter keeps us out of dead-range markets (where BTC loses 77% in 2022)
- 1d SMA200 ensures we don't fight the major trend
- Trailing stop locks profits in volatile breakouts

TARGET: 60-120 total trades over 4 years (15-30/year). HARD MAX: 200.
Signal size: 0.30 (balanced risk).

The key difference from failed #007-#019: BB Squeeze fires LESS than Donchian
(only when volatility contracts first), reducing false breakouts and fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_vol_atr_1d_v1"
timeframe = "4h"
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

def calculate_bollinger_bands(close, period=20, num_std=2):
    """Bollinger Bands - returns (upper, middle, lower, width)"""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    
    return upper, middle, lower, width

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_ratio = atr_14 / np.where(atr_30 > 0, atr_30, 1)
    
    # Bollinger Bands (20,2)
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, num_std=2)
    
    # BB Width percentile (100-bar lookback)
    bb_width_percentile = np.full(n, np.nan, dtype=np.float64)
    for i in range(100, n):
        window = bb_width[i-100:i]
        valid = window[~np.isnan(window)]
        if len(valid) >= 50:
            min_w = np.min(valid)
            max_w = np.max(valid)
            if max_w > min_w:
                bb_width_percentile[i] = (bb_width[i] - min_w) / (max_w - min_w)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ADX for trend strength
    adx = calculate_adx(high, low, close, period=14)
    
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
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + 50 for BB percentile
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(bb_width_percentile[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_200_aligned[i]
        trend_bull = price_above_1d_sma
        trend_bear = not price_above_1d_sma
        
        # === SQUEEZE DETECTION ===
        # BB Width in lower 20th percentile = true compression
        is_squeezed = bb_width_percentile[i] < 0.20
        
        # Volatility expansion (ATR ratio > 1.0 means volatility is rising)
        vol_expansion = atr_ratio[i] > 1.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === TREND STRENGTH (ADX) ===
        adx_strong = not np.isnan(adx[i]) and adx[i] > 20
        
        # === BREAKOUT DETECTION ===
        prev_close = close[i - 1] if i > 0 else close[i]
        current_high = high[i]
        current_low = low[i]
        
        # Long breakout: close above BB upper on squeeze
        long_breakout = close[i] > bb_upper[i] and prev_close <= bb_upper[i - 1] if i > 0 else False
        
        # Short breakdown: close below BB lower on squeeze
        short_breakdown = close[i] < bb_lower[i] and prev_close >= bb_lower[i - 1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Squeeze breakout + bull trend ===
            # Requires: squeezed, vol expansion, volume spike, bull trend
            if is_squeezed and vol_expansion and long_breakout and trend_bull:
                desired_signal = SIZE
            
            # === SHORT ENTRY: Squeeze breakdown + bear trend ===
            # Requires: squeezed, vol expansion, volume spike, bear trend
            if is_squeezed and vol_expansion and short_breakdown and trend_bear:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit if price reverts to BB middle
            if position_side > 0 and close[i] < bb_middle[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > bb_middle[i]:
                desired_signal = 0.0
        
        # === SQUEEZE RESOLUTION EXIT ===
        # If squeeze fully resolves (width returns above median), exit
        if in_position and not np.isnan(bb_width_percentile[i]):
            if bb_width_percentile[i] > 0.60:
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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