#!/usr/bin/env python3
"""
Experiment #026: 12h Camarilla Pivot + Choppiness Regime + Volume Spike

HYPOTHESIS: Camarilla pivot levels (S3/R3) act as strong support/resistance
where price reverses. Combined with Choppiness Index < 61.8 (trending regime)
and volume spikes, this filters out ranging markets and catches institutional
reversal points. Works in both bull (long R3 breaks) and bear (short S3 breaks).

TIMEFRAME: 12h primary
HTF: 1d for trend, 1w for regime
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_chop_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla_pivots(high, low, close, offset=0):
    """
    Camarilla pivots based on yesterday's range.
    S1 = close + (high - low) * 1.1 / 12
    S2 = close + (high - low) * 1.1 / 6
    S3 = close + (high - low) * 1.1 / 4  <- key level
    S4 = close + (high - low) * 1.1 / 2
    
    R1 = close - (high - low) * 1.1 / 12
    R2 = close - (high - low) * 1.1 / 6
    R3 = close - (high - low) * 1.1 / 4  <- key level
    R4 = close - (high - low) * 1.1 / 2
    """
    n = len(close)
    range_val = high - low
    
    S3 = close - range_val * 1.1 / 4
    R3 = close + range_val * 1.1 / 4
    
    return S3, R3

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
    Choppiness Index (CHOP)
    < 38.2 = trending
    > 61.8 = ranging
    Use as regime filter
    """
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                    abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                    abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        range_sum = highest_high - lowest_low
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(period))
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d HMA for trend alignment
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1w HMA for regime (bull/bear filter)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # === Calculate local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Camarilla pivots - use previous bar's high/low/close
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    S3, R3 = calculate_camarilla_pivots(prev_high, prev_low, prev_close)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
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
    cooldown_bars = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Cooldown after exit
        if cooldown_bars > 0:
            cooldown_bars -= 1
        
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        # Choppiness < 61.8 = trending (good for entries)
        # Choppiness > 61.8 = ranging (avoid)
        is_trending = chop[i] < 61.8
        
        # === TREND DIRECTION (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_above_1w_hma = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === RSI VALUE ===
        rsi_val = rsi[i]
        
        # === CAMARILLA LEVELS ===
        s3 = S3[i]
        r3 = R3[i]
        
        # === ENTRY CONDITIONS ===
        # Key: price touches S3/R3 + trending + volume spike + trend aligned
        desired_signal = 0.0
        
        if not in_position and cooldown_bars == 0:
            # === LONG: Price at S3 support + bullish setup ===
            # Price within 0.5% of S3 (bounce zone)
            if not np.isnan(s3) and s3 > 0:
                near_s3 = (close[i] - s3) / s3 < 0.005  # within 0.5%
                
                if near_s3 and is_trending and vol_spike and price_above_1d_hma:
                    # Long bounce from S3 support
                    desired_signal = SIZE
                    cooldown_bars = 6  # 12h * 6 = 3 days cooldown
            
            # === SHORT: Price at R3 resistance + bearish setup ===
            if not np.isnan(r3) and r3 > 0:
                near_r3 = (r3 - close[i]) / r3 < 0.005  # within 0.5%
                
                if near_r3 and is_trending and vol_spike and not price_above_1d_hma:
                    # Short rejection at R3 resistance
                    desired_signal = -SIZE
                    cooldown_bars = 6
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT: Price reached opposite Camarilla level ===
        if in_position and position_side > 0:
            # Take profit if price reaches R3
            if not np.isnan(r3) and r3 > 0:
                if close[i] >= r3:
                    desired_signal = SIZE / 2  # Half position take profit
                    in_position = False  # Exit
        
        if in_position and position_side < 0:
            # Take profit if price reaches S3
            if not np.isnan(s3) and s3 > 0:
                if close[i] <= s3:
                    desired_signal = -SIZE / 2  # Half position take profit
                    in_position = False  # Exit
        
        # === EXIT: Choppiness regime change ===
        if in_position and position_side > 0:
            if not is_trending and rsi_val > 55:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if not is_trending and rsi_val < 45:
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
                # Same direction - maintain position
                pass
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