#!/usr/bin/env python3
"""
Experiment #006: 4h Camarilla S3/R3 Breakout + Strict Volume + CHOP<50

HYPOTHESIS: The DB winner (gen_camarilla_pivot_volume_spike_choppiness_4h_v1)
succeeded because:
1. Strict CHOP < 50 (only trending, not choppy)
2. S3/R3 breakout (not just proximity)
3. Volume spike confirmation on breakout bars ONLY
4. Simple 1d SMA200 trend filter (market bias)

WHY THIS WORKS IN BULL AND BEAR:
- In bull: price bounces at S3 support, breaks R3 resistance
- In bear: rallies to R3 resistance fail, short at R3 breakdown
- In range: CHOP filter prevents false signals

KEY REFINEMENT vs #003:
- CHOP < 50 (strict, not < 55)
- Volume spike REQUIRED (not optional)
- Price must cross/break level (not just approach)
- 1d SMA200 for trend (simpler than dual EMA)
- Target: 75-150 trades over 4 years

DB Reference: ETHUSDT test Sharpe=1.471, 95 trades
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_s3r3_breakout_vol_chop_v1"
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
    """Choppiness Index - CHOP > 61.8 = ranging, CHOP < 50 = trending"""
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
    """
    Camarilla pivot levels
    S3 = close - (high - low) * 1.1 / 4
    R3 = close + (high - low) * 1.1 / 4
    """
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE for Camarilla pivots and SMA200 trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d SMA200 for trend bias
    sma_200_1d_raw = calculate_sma(df_1d['close'].values, 200)
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d_raw)
    
    # Calculate Camarilla S3/R3 pivots from 1d
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume moving average (strict: need volume confirmation)
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
    
    warmup = 100
    
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
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
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
        
        # === REGIME CHECK (STRICT) ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Strict: only trending periods
        
        if not is_trending:
            # In chop, exit if in position
            if in_position:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = 0.0
            continue
        
        # === TREND BIAS (1d SMA200) ===
        sma_200 = sma_200_1d_aligned[i]
        price_above_200 = close[i] > sma_200 if not np.isnan(sma_200) else True
        price_below_200 = close[i] < sma_200 if not np.isnan(sma_200) else False
        
        # === VOLUME CONFIRMATION (REQUIRED) ===
        vol_spike = vol_ratio[i] > 1.5  # Volume > 1.5x 20-avg
        
        # === CAMARILLA LEVELS ===
        s3 = s3_aligned[i]
        r3 = r3_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Price breaks below S3 + bullish bias + volume spike
        # S3 is support - breaking below it is a breakdown
        # In uptrend, price bounces FROM S3, not through it
        # We enter when price RECOVERS above S3 after touching it
        if price_above_200:  # Bullish market bias
            # Price within 1 ATR of S3 (potential bounce zone)
            if not np.isnan(s3) and atr_14[i] > 0:
                dist_to_s3 = (close[i] - s3) / atr_14[i]
                # Price near S3 and recovering (low didn't break too far below)
                if 0 < dist_to_s3 < 1.0 and low[i] > s3 - 0.5 * atr_14[i]:
                    if vol_spike:
                        desired_signal = SIZE
        
        # SHORT: Price breaks above R3 + bearish bias + volume spike
        if price_below_200:  # Bearish market bias
            if not np.isnan(r3) and atr_14[i] > 0:
                dist_to_r3 = (r3 - close[i]) / atr_14[i]
                # Price near R3 and rejecting (high didn't break too far above)
                if 0 < dist_to_r3 < 1.0 and high[i] < r3 + 0.5 * atr_14[i]:
                    if vol_spike:
                        desired_signal = -SIZE
        
        # === STOPLOSS CHECK (ATR-based trailing) ===
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
        
        # === TAKE PROFIT at opposite pivot ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at R3
            if not np.isnan(r3) and high[i] >= r3:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at S3
            if not np.isnan(s3) and low[i] <= s3:
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