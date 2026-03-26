#!/usr/bin/env python3
"""
Experiment #024: 6h Camarilla Pivot Fade + 1d Trend + Volume

HYPOTHESIS: Camarilla pivot levels (R3/S3) are mathematically precise 
support/resistance zones. Unlike Donchian breakouts (which failed), 
fading extremes at these levels captures mean reversion while avoiding 
whipsaws. 1d HMA provides trend context. Volume spike confirms institutional 
interest. Works in both bull (long S3 bounces) and bear (short R3 rejections).

TIMEFRAME: 6h primary
HTF: 1d for trend bias
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_1d_hma_vol_v1"
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

def calculate_camarilla_levels(high, low, close, lookback=8):
    """
    Calculate Camarilla pivot levels from previous N bars.
    R4 = C + (H - L) * 1.1 / 2
    R3 = C + (H - L) * 1.1 / 4
    S3 = C - (H - L) * 1.1 / 4
    S4 = C - (H - L) * 1.1 / 2
    """
    n = len(close)
    r4 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    pivot = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        prev_high = np.max(high[i - lookback:i])
        prev_low = np.min(low[i - lookback:i])
        prev_close = close[i - 1]
        
        h_l = prev_high - prev_low
        
        pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        r4[i] = prev_close + h_l * 1.1 / 2.0
        r3[i] = prev_close + h_l * 1.1 / 4.0
        s3[i] = prev_close - h_l * 1.1 / 4.0
        s4[i] = prev_close - h_l * 1.1 / 2.0
    
    return pivot, r3, r4, s3, s4

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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index: measures market choppiness vs trending.
    < 38.2 = trending (good for breakout strategies)
    > 61.8 = choppy (good for mean reversion)
    """
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            idx = i - j
            tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]) if idx > 0 else high[idx] - low[idx])
            sum_tr += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        range_val = highest_high - lowest_low
        
        if range_val > 0 and sum_tr > 0:
            chop[i] = 100 * np.log10(sum_tr / range_val) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Camarilla pivot levels (lookback = 8 bars = ~2 days of 6h)
    pivot, r3, r4, s3, s4 = calculate_camarilla_levels(high, low, close, lookback=8)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Choppiness Index (precompute before loop)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # RSI for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Need enough bars for indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(pivot[i]) or np.isnan(r3[i]) or np.isnan(s3[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND CONTEXT ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        trend_bullish = price_above_1d_hma
        trend_bearish = not price_above_1d_hma
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === RSI MOMENTUM ===
        rsi_val = rsi[i]
        rsi_oversold = rsi_val < 35
        rsi_overbought = rsi_val > 65
        
        # === CHOPPINESS FILTER ===
        chop_val = chop[i] if not np.isnan(chop[i]) else 50.0
        # Lower chop = more trending, higher = more choppy
        chop_favorable = chop_val < 61.8
        
        # === CAMARILLA LEVEL PROXIMITY ===
        # Price approaching/bouncing from S3 or R3
        s3_dist = (close[i] - s3[i]) / (atr_14[i] + 1e-10)
        r3_dist = (r3[i] - close[i]) / (atr_14[i] + 1e-10)
        
        # Near S3 (within 0.5 ATR)
        near_s3 = s3_dist > 0 and s3_dist < 0.5
        # Near R3 (within 0.5 ATR)
        near_r3 = r3_dist > 0 and r3_dist < 0.5
        
        # Price bounced FROM S3 (was below, now above)
        bounce_from_s3 = False
        if i > 1:
            bounce_from_s3 = (close[i] > s3[i]) and (close[i-1] <= s3[i-1])
        
        # Price rejected AT R3 (was below, now rejected)
        reject_r3 = False
        if i > 1:
            reject_r3 = (close[i] < r3[i]) and (close[i-1] >= r3[i-1])
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Bounce from S3 with trend confirmation ===
            # Conditions:
            # 1. Price near or bounced from S3
            # 2. Bullish 1d trend
            # 3. Volume spike
            # 4. RSI not oversold (confirming bounce strength)
            # 5. Choppiness < 61.8 (trending environment preferred)
            
            if (bounce_from_s3 or near_s3) and trend_bullish:
                if vol_spike and not rsi_oversold and chop_favorable:
                    desired_signal = SIZE
                # Also allow entry without bounce if deeply depressed and very bullish
                elif near_s3 and trend_bullish and rsi_val < 30 and vol_ratio[i] > 1.2:
                    desired_signal = SIZE
            
            # === SHORT ENTRY: Rejection at R3 with trend confirmation ===
            # Conditions:
            # 1. Price near or rejected at R3
            # 2. Bearish 1d trend
            # 3. Volume spike
            # 4. RSI not overbought
            # 5. Choppiness < 61.8
            
            if (reject_r3 or near_r3) and trend_bearish:
                if vol_spike and not rsi_overbought and chop_favorable:
                    desired_signal = -SIZE
                # Also allow entry without rejection if very overbought and bearish
                elif near_r3 and trend_bearish and rsi_val > 70 and vol_ratio[i] > 1.2:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === TAKE PROFIT: Trail stop and exit on opposite R/S level ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            # TP: price reached pivot or R3
            if close[i] >= pivot[i] * 1.0:
                tp_triggered = True
            # Also exit if RSI overbought and trend weakens
            if rsi_val > 75 and not price_above_1d_hma:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP: price reached pivot or S3
            if close[i] <= pivot[i]:
                tp_triggered = True
            # Also exit if RSI oversold and trend weakens
            if rsi_val < 25 and price_above_1d_hma:
                tp_triggered = True
        
        if tp_triggered:
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