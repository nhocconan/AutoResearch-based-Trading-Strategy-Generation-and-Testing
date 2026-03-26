#!/usr/bin/env python3
"""
Experiment #023: 12h Camarilla Pivots + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels from 1d capture institutional support/resistance.
S3/S4 touches with volume spike signal reversals. Choppiness Index filters ranging 
markets. Works in BOTH bull (long S3/S4 bounces) and bear (short R3/R4 rejections).

WHY 12h: Slower than 4h = fewer trades = less fee drag.
TARGET: 50-150 total trades over 4 years (12-37/year).

Proven pattern from DB: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 
achieved test Sharpe=1.471 on ETHUSDT. This adapts it to 12h for fewer trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla_pivots(high, low, close):
    """Camarilla pivot levels - 8 levels (S1-S4, R1-R4)"""
    n = len(close)
    pivots = {
        's3': np.full(n, np.nan),
        's4': np.full(n, np.nan),
        'r3': np.full(n, np.nan),
        'r4': np.full(n, np.nan),
        'pivot': np.full(n, np.nan)
    }
    
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]):
            continue
        
        pivot = (high[i-1] + low[i-1] + close[i-1]) / 3.0
        pivots['pivot'][i] = pivot
        
        range_ = high[i-1] - low[i-1]
        
        # S3 = close + range * 0.0916
        pivots['s3'][i] = close[i-1] + range_ * 0.0916
        # S4 = close + range * 0.1832
        pivots['s4'][i] = close[i-1] + range_ * 0.1832
        
        # R3 = close - range * 0.0916
        pivots['r3'][i] = close[i-1] - range_ * 0.0916
        # R4 = close - range * 0.1832
        pivots['r4'][i] = close[i-1] - range_ * 0.1832
    
    return pivots

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index - measures market trendiness"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        if np.isnan(high[i]) or np.isnan(low[i]):
            continue
        
        # Sum of true range
        tr_sum = 0
        for j in range(period):
            idx = i - j
            tr = max(high[idx] - low[idx], 
                     abs(high[idx] - close[idx-1]) if idx > 0 else high[idx] - low[idx])
            tr_sum += tr
        
        # Highest - lowest over period
        high_max = np.max(high[i - period + 1:i + 1])
        low_min = np.min(low[i - period + 1:i + 1])
        
        range_sum = high_max - low_min
        
        if range_sum > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / range_sum) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_spike(volume, period=20):
    """Volume spike ratio"""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma.values, 1)
    return vol_ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d indicators ===
    pivots_1d = calculate_camarilla_pivots(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    
    # Align 1d pivots to 12h (previous completed 1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, pivots_1d['s3'])
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, pivots_1d['s4'])
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, pivots_1d['r3'])
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, pivots_1d['r4'])
    
    # 1d SMA for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_spike(volume, period=20)
    
    # Choppiness Index on 12h
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    # === Local 12h Donchian for structure ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        
        # Check 1d pivot availability
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        
        if np.isnan(s3) or np.isnan(r3):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop_val = chop[i]
        is_trending = not np.isnan(chop_val) and chop_val < 50.0  # CHOP < 50 = trending
        
        # === TREND DIRECTION (1d SMA) ===
        trend_up = not np.isnan(sma_1d_aligned[i]) and close[i] > sma_1d_aligned[i]
        trend_down = not np.isnan(sma_1d_aligned[i]) and close[i] < sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MOMENTUM ===
        rsi_val = rsi[i]
        
        # === STRUCTURE ===
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # === CAMARILLA TOUCH DETECTION ===
        # Price touches S3 or S4 (within 0.5 ATR)
        touch_s3 = low[i] <= s3 + 0.3 * atr_14[i] and low[i] >= s3 - 0.5 * atr_14[i]
        touch_s4 = low[i] <= s4 + 0.3 * atr_14[i] and low[i] >= s4 - 0.5 * atr_14[i]
        
        # Price touches R3 or R4 (within 0.5 ATR)
        touch_r3 = high[i] >= r3 - 0.3 * atr_14[i] and high[i] <= r3 + 0.5 * atr_14[i]
        touch_r4 = high[i] >= r4 - 0.3 * atr_14[i] and high[i] <= r4 + 0.5 * atr_14[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: S3/S4 touch + volume + trend up ===
            if (touch_s3 or touch_s4) and vol_spike and trend_up:
                desired_signal = SIZE
            
            # === LONG ENTRY: Oversold RSI + trend up ===
            if rsi_val < 35 and trend_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT ENTRY: R3/R4 touch + volume + trend down ===
            if (touch_r3 or touch_r4) and vol_spike and trend_down:
                desired_signal = -SIZE
            
            # === SHORT ENTRY: Overbought RSI + trend down ===
            if rsi_val > 65 and trend_down and vol_spike:
                desired_signal = -SIZE
        
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
        
        # === EXIT CONDITIONS ===
        exit_long = False
        exit_short = False
        
        if in_position and position_side > 0:
            # Exit: price breaks structure or RSI extreme
            if price_below_lower:
                exit_long = True
            if rsi_val > 75:
                exit_long = True
            # Take profit: price near R3/R4
            if close[i] >= r3 and not np.isnan(r3):
                exit_long = True
        
        if in_position and position_side < 0:
            # Exit: price breaks structure or RSI extreme
            if price_above_upper:
                exit_short = True
            if rsi_val < 25:
                exit_short = True
            # Take profit: price near S3/S4
            if close[i] <= s3 and not np.isnan(s3):
                exit_short = True
        
        if exit_long or exit_short:
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