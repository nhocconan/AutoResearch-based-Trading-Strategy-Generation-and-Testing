#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + ADX Regime Filter with Volume Spike Confirmation
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
- ADX > 25 indicates trending market (use Elder Ray for direction)
- ADX < 20 indicates ranging market (fade extreme Elder Ray readings)
- Volume Spike: Current volume > 2.0x 20-period average for confirmation
- Uses 12h HTF for regime classification to avoid whipsaws
- Target: 12-25 trades/year per symbol (~50-100 total over 4 years)
- Position sizing: 0.25 (discrete levels to minimize fee churn)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for primary calculations
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Get 12h data for regime filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA13 on 6h for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components on 6h
    bull_power = high_6h - ema13_6h
    bear_power = ema13_6h - low_6h
    
    # Calculate ADX on 12h for regime filter
    def calculate_adx(high_arr, low_arr, close_arr, window=14):
        """Average Directional Index"""
        # True Range
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        dm_plus = np.where((high_arr - np.roll(high_arr, 1)) > (np.roll(low_arr, 1) - low_arr),
                           np.maximum(high_arr - np.roll(high_arr, 1), 0), 0)
        dm_minus = np.where((np.roll(low_arr, 1) - low_arr) > (high_arr - np.roll(high_arr, 1)),
                            np.maximum(np.roll(low_arr, 1) - low_arr, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_smoothed = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        dm_plus_smoothed = pd.Series(dm_plus).ewm(span=window, adjust=False, min_periods=window).mean().values
        dm_minus_smoothed = pd.Series(dm_minus).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smoothed / tr_smoothed
        di_minus = 100 * dm_minus_smoothed / tr_smoothed
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx[np.isnan(dx)] = 0
        adx = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Volume average (20-period) on 6h
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss on 6h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_6h = calculate_atr(high_6h, low_6h, close_6h, 14)
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        adx = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Regime-based entry logic
            if adx > 25:  # Trending market - use Elder Ray for direction
                # Long: Strong bull power + volume spike
                if bull > 0 and bull > bear and vol > 2.0 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: Strong bear power + volume spike
                elif bear > 0 and bear > bull and vol > 2.0 * vol_ma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            else:  # Ranging market (ADX < 25) - fade extreme readings
                # Long: Oversold condition (extreme bear power) + volume spike
                if bear < 0 and abs(bear) > abs(bull) and vol > 2.0 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: Overbought condition (extreme bull power) + volume spike
                elif bull < 0 and abs(bull) > abs(bear) and vol > 2.0 * vol_ma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Elder Ray divergence (bull power weakening)
            if bull < 0:  # Bull power turned negative
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.5 * ATR below entry)
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Elder Ray divergence (bear power weakening)
            if bear < 0:  # Bear power turned negative
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.5 * ATR above entry)
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0