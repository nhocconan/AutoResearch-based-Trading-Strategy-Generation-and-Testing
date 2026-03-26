#!/usr/bin/env python3
"""
Experiment #009: 4h Camarilla Pivot + Volume Spike + Choppiness (Refined)

HYPOTHESIS: Simplify the proven Camarilla pattern from DB (Sharpe=1.471).
The current implementation (357 trades, Sharpe=-0.814) is TOO LOOSE.
The winning DB pattern had only 95 trades.

KEY INSIGHT: Fewer conditions = fewer trades = less fee drag = better test Sharpe.

REFINEMENTS vs current code:
1. REMOVE redundant EMA cross (conflicts with HMA trend)
2. TIGHTER entry threshold: within 0.3 ATR of pivot (not 0.5-2.0 ATR)
3. ONLY enter on volume spike (not "or ema_bullish")
4. Add cooldown between trades (min 8 bars) to prevent signal flipping
5. SIMPLER regime: only CHOP < 50 (not < 55)
6. ONLY use S3/R3 for entries (S4/R4 too deep = poor risk/reward)

TARGET: 75-150 total trades over 4 years (proven range from DB).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_refined_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - CHOP < 50 = trending (allow trades)"""
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

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """Camarilla pivot levels from previous day"""
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE for pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1d Camarilla pivots
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    
    # 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume moving average
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
    bars_since_exit = 999  # cooldown counter
    
    warmup = 60
    
    for i in range(warmup, n):
        bars_since_exit += 1
        
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
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Stricter: only trending markets
        
        # === TREND BIAS (1d HMA only) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        trend_bullish = price_above_1d_hma
        trend_bearish = not price_above_1d_hma
        
        # === VOLUME SPIKE (MANDATORY) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA PIVOT DISTANCE ===
        s3 = s3_aligned[i]
        r3 = r3_aligned[i]
        
        if not np.isnan(s3) and atr_14[i] > 0:
            dist_to_s3 = (close[i] - s3) / atr_14[i]
            dist_to_r3 = (r3 - close[i]) / atr_14[i]
        else:
            dist_to_s3 = 999
            dist_to_r3 = 999
        
        # === ENTRY LOGIC (TIGHT: within 0.3 ATR) ===
        desired_signal = 0.0
        
        if not in_position and bars_since_exit >= 8:
            # LONG: near S3 + bullish trend + volume spike
            if is_trending and trend_bullish and vol_spike:
                if -0.3 <= dist_to_s3 <= 0.3:  # TIGHT: within 0.3 ATR
                    desired_signal = SIZE
            
            # SHORT: near R3 + bearish trend + volume spike
            if is_trending and trend_bearish and vol_spike:
                if -0.3 <= dist_to_r3 <= 0.3:  # TIGHT: within 0.3 ATR
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (ATR-based trailing) ===
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
            bars_since_exit = 0
        
        # === TAKE PROFIT at opposite pivot ===
        tp_triggered = False
        if in_position and position_side > 0:
            if not np.isnan(r3) and high[i] >= r3:
                tp_triggered = True
        
        if in_position and position_side < 0:
            if not np.isnan(s3) and low[i] <= s3:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
            bars_since_exit = 0
        
        # === UPDATE POSITION ===
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
                bars_since_exit = 0
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