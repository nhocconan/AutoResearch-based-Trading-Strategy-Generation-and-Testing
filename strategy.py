#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian Breakout + Williams %R + 1d SMA200

HYPOTHESIS: Price breaking above 20-period Donchian high on 4h captures
momentum shifts. Using 1d SMA200 for trend filter avoids false breakouts
in downtrends. Williams %R confirms overbought/oversold conditions at breakout.

This works in BOTH bull and bear:
- Bull: Buy breakout above SMA200 with Williams %R confirming
- Bear: Short breakdown below SMA200 with Williams %R confirming

TARGET: 75-200 total trades over 4 years (19-50/year). HARD MAX: 400.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_willr_sma200_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range using EWM"""
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

def calculate_willr(high, low, close, period=14):
    """Williams %R: -100 * (HH - Close) / (HH - LL)"""
    n = len(close)
    willr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        if highest_high > lowest_low:
            willr[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    willr_14 = calculate_willr(high, low, close, period=14)
    
    # Donchian(20) for breakout detection
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    prev_close = close[0]
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        prev_close = close[i - 1]
        
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        trend_up = close[i] > sma_1d_aligned[i]
        trend_down = close[i] < sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === WILLIAMS %R MOMENTUM ===
        willr_oversold = willr_14[i] < -80  # Bullish momentum
        willr_overbought = willr_14[i] > -20  # Bearish momentum
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Breakout above Donchian(20) high + SMA200 trend up + Williams %R confirm + volume
            if trend_up and willr_oversold and vol_spike:
                if high[i] > donch_high[i - 1]:  # New 20-bar high
                    desired_signal = SIZE
            
            # SHORT: Breakdown below Donchian(20) low + SMA200 trend down + Williams %R confirm + volume
            if trend_down and willr_overbought and vol_spike:
                if low[i] < donch_low[i - 1]:  # New 20-bar low
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT (mid-channel within 8 bars) ===
        bars_held = i - entry_bar
        
        if in_position and 2 <= bars_held <= 8:
            # Long: close near mid-channel = take profit
            if position_side > 0 and close[i] >= donch_mid[i - 1]:
                desired_signal = 0.0
            # Short: close near mid-channel = take profit
            if position_side < 0 and close[i] <= donch_mid[i - 1]:
                desired_signal = 0.0
        
        # === TREND EXIT (if SMA200 flips) ===
        if in_position and bars_held >= 2:
            if position_side > 0 and not trend_up:
                desired_signal = 0.0
            if position_side < 0 and not trend_down:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals