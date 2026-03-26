#!/usr/bin/env python3
"""
Experiment #023: 12h Camarilla + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels mark institutional support/resistance on 12h TF.
Combined with volume spike confirmation (1.5x) and Choppiness Index regime filter,
this captures major moves while avoiding whipsaws. Choppiness < 50 = trending
(follow breakouts), Choppiness > 50 = ranging (skip or fade).

TIMEFRAME: 12h primary
HTF: 1w for regime, 1d for trend
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_regime_v1"
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

def calculate_choppiness(close, period=14):
    """Choppiness Index - values > 61.8 = choppy, < 38.2 = trending"""
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if np.isnan(close[i]):
            continue
        
        period_high = np.max(close[i - period + 1:i + 1])
        period_low = np.min(close[i - period + 1:i + 1])
        
        range_size = period_high - period_low
        if range_size <= 0:
            continue
        
        atr_sum = 0.0
        valid = True
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        if atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_size) / np.log10(period)
    
    return chop

def calculate_camarilla_pivots(high, low, close, period=20):
    """Camarilla pivot levels - returns arrays for S3, S4, R3, R4"""
    n = len(close)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        pivot = (high[i] + low[i] + close[i]) / 3.0
        range_val = high[i] - low[i]
        
        # Camarilla levels
        s3[i] = close[i] + range_val * 0.1
        s4[i] = close[i] + range_val * 0.183
        r3[i] = close[i] - range_val * 0.1
        r4[i] = close[i] - range_val * 0.183
    
    return s3, s4, r3, r4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # 1w HMA for regime (bull/bear/range)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # 1d HMA for trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(close, period=14)
    
    # Camarilla pivots
    s3, s4, r3, r4 = calculate_camarilla_pivots(high, low, close, period=20)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Williams %R for momentum
    willr = np.full(n, np.nan, dtype=np.float64)
    period_wr = 14
    for i in range(period_wr, n):
        highest = np.max(high[i - period_wr + 1:i + 1])
        lowest = np.min(low[i - period_wr + 1:i + 1])
        range_wr = highest - lowest
        if range_wr > 0:
            willr[i] = -100 * (highest - close[i]) / range_wr
    
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
    
    warmup = 50
    
    for i in range(warmup, n):
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (1w HMA) ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        
        # === TREND (1d HMA) ===
        daily_trend_up = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else weekly_bullish
        
        # === CHOPPINESS REGIME ===
        # CHOP < 50 = trending (good for breakouts)
        # CHOP > 50 = ranging (skip breakout entries)
        trending = chop[i] < 50.0
        very_trending = chop[i] < 38.2
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MOMENTUM (Williams %R) ===
        willr_val = willr[i]
        oversold = willr_val < -80
        overbought = willr_val > -20
        
        # === CAMARILLA LEVEL TOUCH ===
        # Price touches/fills S3 or R3 with confirmation
        s3_touched = low[i] <= s3[i] if not np.isnan(s3[i]) else False
        r3_touched = high[i] >= r3[i] if not np.isnan(r3[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # S3 touched + oversold + volume spike + bullish regime
            if s3_touched and oversold and vol_spike and weekly_bullish:
                # In ranging markets, require stronger confluence
                if trending:
                    desired_signal = SIZE
                elif very_trending and daily_trend_up:
                    # Only in very trending + aligned
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # R3 touched + overbought + volume spike + bearish regime
            if r3_touched and overbought and vol_spike and not weekly_bullish:
                if trending:
                    desired_signal = -SIZE
                elif very_trending and not daily_trend_up:
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
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: RSI overbought or opposite signal
            if r3_touched and overbought:
                exit_triggered = True
            # Time-based exit after 8 bars if profitable
            if (i - entry_bar) > 8 and willr_val > -30:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: RSI oversold or opposite signal
            if s3_touched and oversold:
                exit_triggered = True
            if (i - entry_bar) > 8 and willr_val < -70:
                exit_triggered = True
        
        if exit_triggered:
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
                entry_bar = i
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
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals