#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ATR expansion + Donchian(20) breakout with volume spike confirmation and ATR trailing stop.
# Long when price breaks above Donchian(20) high AND 1d ATR(7)/ATR(30) > 1.5 AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d ATR(7)/ATR(30) > 1.5 AND volume > 1.5x 20-period average.
# Exit via ATR trailing stop (2.5 * ATR) or opposite Donchian breakout.
# Uses discrete position size 0.25. Targets 12-37 trades/year to minimize fee drag.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns) with volatility filter.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ATR(7) and ATR(30) for volatility expansion filter ===
    def atr(high, low, close, period):
        """Average True Range"""
        if len(high) < period:
            return np.full(len(high), np.nan)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]  # First TR
        atr_vals = np.full(len(high), np.nan)
        atr_vals[period-1] = np.mean(tr[:period])
        for i in range(period, len(high)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr7_1d = atr(high_1d, low_1d, close_1d, 7)
    atr30_1d = atr(high_1d, low_1d, close_1d, 30)
    
    # === Primary timeframe (12h) indicators: Donchian(20) channels ===
    def donchian_channels(high, low, period):
        """Donchian channels: upper = max(high, period), lower = min(low, period)"""
        if len(high) < period:
            return np.full(len(high), np.nan), np.full(len(high), np.nan)
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    atr7_aligned = align_htf_to_ltf(prices, df_1d, atr7_1d)
    atr30_aligned = align_htf_to_ltf(prices, df_1d, atr30_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # Donchian20 + ATR30 + volume MA20
    
    # Track position state and ATR for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr7_aligned[i]) or np.isnan(atr30_aligned[i]) or 
            np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        atr7 = atr7_aligned[i]
        atr30 = atr30_aligned[i]
        atr_ratio = atr7 / atr30 if atr30 > 0 else 0
        upper = upper_20[i]
        lower = lower_20[i]
        vol_ma = vol_ma_20[i]
        price = close[i]
        vol_spike = volume[i] > (1.5 * vol_ma)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # ATR trailing stop: exit if price drops 2.5*ATR from highest high since entry
            if price < (highest_high_since_entry - 2.5 * atr30):
                exit_signal = True
            # Opposite Donchian breakout: exit if price breaks below lower Donchian
            elif price < lower:
                exit_signal = True
        
        elif position == -1:  # Short position
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest low since entry
            if price > (lowest_low_since_entry + 2.5 * atr30):
                exit_signal = True
            # Opposite Donchian breakout: exit if price breaks above upper Donchian
            elif price > upper:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            highest_high_since_entry = 0
            lowest_low_since_entry = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volatility expansion filter: ATR(7)/ATR(30) > 1.5 indicates expanding volatility
            vol_expansion = atr_ratio > 1.5
            
            # LONG: Price breaks above upper Donchian AND volume spike AND volatility expansion
            if (price > upper) and vol_spike and vol_expansion:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
                lowest_low_since_entry = price
            
            # SHORT: Price breaks below lower Donchian AND volume spike AND volatility expansion
            elif (price < lower) and vol_spike and vol_expansion:
                signals[i] = -0.25
                position = -1
                highest_high_since_entry = price
                lowest_low_since_entry = price
        
        else:
            # Maintain position and update highest/lowest for trailing stop
            signals[i] = position * 0.25
            if position == 1:
                highest_high_since_entry = max(highest_high_since_entry, price)
                lowest_low_since_entry = min(lowest_low_since_entry, price)
            elif position == -1:
                highest_high_since_entry = max(highest_high_since_entry, price)
                lowest_low_since_entry = min(lowest_low_since_entry, price)
    
    return signals

name = "12h_1dATRExpansion_Donchian20_Breakout_VolumeSpike_ATRTrail2.5_V1"
timeframe = "12h"
leverage = 1.0