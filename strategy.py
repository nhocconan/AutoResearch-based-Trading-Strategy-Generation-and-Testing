#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h Supertrend(ATR=10, mult=3) trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band + 12h Supertrend = uptrend + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian lower band + 12h Supertrend = downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to balance return and drawdown control.
# 12h Supertrend provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~20-30 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Supertrend(ATR=10, mult=3) ===
    def calculate_atr(high, low, close, period):
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        ranges = np.vstack([high_low, high_close, low_close])
        tr = np.max(ranges, axis=0)
        tr[0] = 0  # First period has no prior close
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean()
        return atr.values
    
    def calculate_supertrend(high, low, close, atr_period=10, multiplier=3):
        """Calculate Supertrend indicator"""
        atr = calculate_atr(high, low, close, atr_period)
        hl2 = (high + low) / 2
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        supertrend = np.zeros_like(close)
        direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
        
        supertrend[0] = 0
        direction[0] = 1
        
        for i in range(1, len(close)):
            if close[i] > upperband[i-1]:
                direction[i] = 1
            elif close[i] < lowerband[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                    lowerband[i] = lowerband[i-1]
                if direction[i] == -1 and upperband[i] > upperband[i-1]:
                    upperband[i] = upperband[i-1]
            
            if direction[i] == 1:
                supertrend[i] = lowerband[i]
            else:
                supertrend[i] = upperband[i]
        
        return supertrend, direction
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    supertrend_12h, direction_12h = calculate_supertrend(high_12h, low_12h, close_12h, 10, 3)
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # === 4h Donchian Channel (20-period) ===
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    upper_band = highest_high
    lower_band = lowest_low
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(period, 20) + 30  # Donchian(20) + volume(20) + Supertrend buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(supertrend_12h_aligned[i]) or np.isnan(direction_12h_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 12h close for trend filter
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper band (close > upper_band)
        # 2. 12h Supertrend uptrend (direction = 1)
        # 3. Volume confirmation
        if (close[i] > upper_band[i]) and \
           (direction_12h_aligned[i] == 1) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower band (close < lower_band)
        # 2. 12h Supertrend downtrend (direction = -1)
        # 3. Volume confirmation
        elif (close[i] < lower_band[i]) and \
             (direction_12h_aligned[i] == -1) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hSupertrend_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0