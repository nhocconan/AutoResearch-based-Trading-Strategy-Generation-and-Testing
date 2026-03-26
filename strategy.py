#!/usr/bin/env python3
"""
Experiment #007: Williams Alligator + ADX on 6h

HYPOTHESIS: Williams Alligator captures institutional "sleeping/awakening/eating" 
phases that standard MAs miss. When the Alligator "wakes up" (Lips crosses Jaw/Teeth),
it signals the start of directional moves. ADX confirms trend strength.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Alligator "sleeps" in range (no trades during chop) — avoids 2022 whipsaw
- "Awakening" = Lips crossing Jaw/Teeth = early trend confirmation
- ADX > 20 confirms trend has momentum before entry
- Symmetric logic for long/short — works in both directions

TARGET: 80-150 total trades over 4 years (19-37/year on 6h).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williams_alligator_adx_1d_v1"
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

def calculate_smma(close, period):
    """Smoothed Moving Average (Williams Alligator)"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    result = np.full(n, np.nan, dtype=np.float64)
    # Initialize with SMA
    sma = np.mean(close[:period])
    result[period - 1] = sma
    
    # SMMA formula: (prev_smma * (period - 1) + current) / period
    for i in range(period, n):
        if not np.isnan(close[i]):
            result[i] = (result[i - 1] * (period - 1) + close[i]) / period
    
    return result

def calculate_alligator(high, low, period_jaw=13, period_teeth=8, period_lips=5):
    """
    Williams Alligator indicator
    Jaw = SMMA(close, 13) — slowest
    Teeth = SMMA(close, 8)
    Lips = SMMA(close, 5) — fastest
    """
    n = len(high)
    if n < period_jaw:
        return {'jaw': np.full(n, np.nan), 'teeth': np.full(n, np.nan), 'lips': np.full(n, np.nan)}
    
    close = (high + low) / 2  # Median price typical for Alligator
    
    jaw = calculate_smma(close, period_jaw)
    teeth = calculate_smma(close, period_teeth)
    lips = calculate_smma(close, period_lips)
    
    return {'jaw': jaw, 'teeth': teeth, 'lips': lips}

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index
    Returns ADX, +DI, -DI
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    plus_di_smooth = np.where(atr > 1e-10, 100 * plus_di / atr, 0)
    minus_di_smooth = np.where(atr > 1e-10, 100 * minus_di / atr, 0)
    
    # DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    valid_idx = (plus_di_smooth + minus_di_smooth) > 1e-10
    dx[valid_idx] = 100 * np.abs(plus_di_smooth[valid_idx] - minus_di_smooth[valid_idx]) / (plus_di_smooth[valid_idx] + minus_di_smooth[valid_idx])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di_smooth, minus_di_smooth

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
    
    # Load 1d data ONCE for trend bias
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    alligator = calculate_alligator(high, low)
    jaw = alligator['jaw']
    teeth = alligator['teeth']
    lips = alligator['lips']
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Moderate size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup - need at least 13 bars for Alligator Jaw
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = True
        if not np.isnan(hma_1d_aligned[i]):
            price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === ADX TREND STRENGTH ===
        adx_strength = adx[i]
        is_trending = adx_strength > 20.0
        
        # === ALLIGATOR CROSSOVER SIGNALS ===
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Need previous values for crossover detection
        if i < warmup + 1:
            signals[i] = 0.0
            continue
        
        jaw_prev = jaw[i - 1]
        teeth_prev = teeth[i - 1]
        lips_prev = lips[i - 1]
        
        # Previous bar states
        prev_bullish = lips_prev > teeth_prev and lips_prev > jaw_prev
        prev_bearish = lips_prev < teeth_prev and lips_prev < jaw_prev
        
        # Current bar states  
        curr_bullish = lips_val > teeth_val and lips_val > jaw_val
        curr_bearish = lips_val < teeth_val and lips_val < jaw_val
        
        # Crossover detection
        bullish_cross = not prev_bullish and curr_bullish
        bearish_cross = not prev_bearish and curr_bearish
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Bullish Alligator crossover + ADX confirms trend + trend bias
        if bullish_cross and is_trending:
            if price_above_1d_hma:  # Only long in uptrend
                if vol_spike:
                    desired_signal = SIZE
                else:
                    desired_signal = SIZE * 0.5  # Reduced if no volume
        
        # SHORT ENTRY: Bearish Alligator crossover + ADX confirms trend + trend bias
        if bearish_cross and is_trending:
            if not price_above_1d_hma:  # Only short in downtrend
                if vol_spike:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.5  # Reduced if no volume
        
        # === STOPLOSS CHECK (2*ATR) ===
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
        
        # === TAKE PROFIT (opposite Alligator signal) ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP when bearish cross occurs (Alligator shows reversal)
            if bearish_cross:
                tp_triggered = True
            # Also TP if price falls below entry - 3*ATR (aggressive exit)
            if close[i] < entry_price - 3.0 * entry_atr:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP when bullish cross occurs (Alligator shows reversal)
            if bullish_cross:
                tp_triggered = True
            # Also TP if price rises above entry + 3*ATR (aggressive exit)
            if close[i] > entry_price + 3.0 * entry_atr:
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