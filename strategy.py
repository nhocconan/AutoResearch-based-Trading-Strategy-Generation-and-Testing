#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian Breakout + Williams %R + 1d EMA200

HYPOTHESIS: Donchian(20) 4h breakout captures institutional momentum shifts.
Williams %R confirms overbought/oversold extremes at the breakout point.
1d EMA200 filters entries to align with higher timeframe trend.
ATR-based stops provide disciplined risk management.

WHY IT WORKS: Simple, robust structure. Breakouts occur in both bull and bear
markets. The combination of price channel breakout + momentum confirmation
reduces false signals. Works in both directions.

TARGET: 100-200 total trades over 4 years. HARD MAX: 300.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_willr_ema200_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr

def calculate_willr(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    if n < period:
        return np.full(n, -50.0)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    willr = np.zeros(n)
    for i in range(period - 1, n):
        hh = highest_high[i]
        ll = lowest_low[i]
        if hh != ll:
            willr[i] = -100 * (hh - close[i]) / (hh - ll)
        else:
            willr[i] = -50.0
    
    return willr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend direction
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === Local 4h indicators ===
    atr = calculate_atr(high, low, close, period=14)
    willr = calculate_willr(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian channels (20 bars = 5 days)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = max(200, donchian_period)  # Need enough for EMA200 and Donchian
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        
        # Skip if HTF EMA not aligned
        if np.isnan(ema200_aligned[i]):
            continue
        
        # Trend direction from 1d EMA200
        bull_trend = close[i] > ema200_aligned[i]
        bear_trend = close[i] < ema200_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Williams %R extremes
        willr_oversold = willr[i] < -80   # Strong bullish momentum
        willr_overbought = willr[i] > -20  # Strong bearish momentum
        
        # Donchian breakout: price closes beyond previous channel
        donchian_upper = donchian_high[i - 1]  # Previous bar's 20-bar high
        donchian_lower = donchian_low[i - 1]   # Previous bar's 20-bar low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if position == 0:
            # LONG: Break above Donchian high + volume + trend + %R confirm
            if bull_trend and vol_spike and willr_oversold:
                if close[i] > donchian_upper:
                    desired_signal = SIZE
            
            # SHORT: Break below Donchian low + volume + trend + %R confirm
            if bear_trend and vol_spike and willr_overbought:
                if close[i] < donchian_lower:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if position > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if position < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === TAKE PROFIT: Williams %R reversal ===
        bars_held = i - entry_bar
        if position > 0 and bars_held >= 3:
            # Take profit if %R shows overbought reversal
            if willr[i] > -20:
                desired_signal = 0.0
        
        if position < 0 and bars_held >= 3:
            # Take profit if %R shows oversold reversal
            if willr[i] < -80:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if position == 0 or np.sign(desired_signal) != position:
                # New position or flip
                position = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        
        if desired_signal == 0.0 and position != 0:
            position = 0
        
        signals[i] = desired_signal
    
    return signals