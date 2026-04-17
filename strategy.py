#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d Elder Ray + volume spike + ATR stoploss.
Long when Alligator bullish (jaw < teeth < lips) AND Elder Bull Power > 0 AND volume > 2.0x 20-period average.
Short when Alligator bearish (jaw > teeth > lips) AND Elder Bear Power < 0 AND volume > 2.0x 20-period average.
Exit when Alligator reverses OR ATR-based stoploss hit (2.5 * ATR).
Uses 1w HTF for regime filter (only trade in bull regime when price > 1w EMA50, bear regime when price < 1w EMA50).
Target: 12-30 trades/year per symbol to minimize fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator (jaw=13, teeth=8, lips=5 SMAs with offsets)
    def calculate_alligator(high_arr, low_arr, close_arr):
        """Williams Alligator: jaw(13,8), teeth(8,5), lips(5,3)"""
        # Typical price
        tp = (high_arr + low_arr + close_arr) / 3.0
        
        # Jaw: 13-period SMMA smoothed 8 bars ahead
        jaw = pd.Series(tp).rolling(window=13, min_periods=13).mean()
        jaw = jaw.shift(8)  # offset 8 bars
        
        # Teeth: 8-period SMMA smoothed 5 bars ahead
        teeth = pd.Series(tp).rolling(window=8, min_periods=8).mean()
        teeth = teeth.shift(5)  # offset 5 bars
        
        # Lips: 5-period SMMA smoothed 3 bars ahead
        lips = pd.Series(tp).rolling(window=5, min_periods=5).mean()
        lips = lips.shift(3)  # offset 3 bars
        
        return jaw.values, teeth.values, lips.values
    
    # Get 1d data for Elder Ray Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    def calculate_elder_ray(high_arr, low_arr, close_arr, ema_period=13):
        """Elder Ray Index"""
        ema = pd.Series(close_arr).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
        bull_power = high_arr - ema
        bear_power = low_arr - ema
        return bull_power, bear_power
    
    # Get 1w data for regime filter (only trade with higher timeframe trend)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate indicators
    jaw, teeth, lips = calculate_alligator(high_12h, low_12h, close_12h)
    bull_power, bear_power = calculate_elder_ray(high_1d, low_1d, close_1d)
    
    # Volume average (20-period) on 12h
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss on 12h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr = calculate_atr(high_12h, low_12h, close_12h, 14)
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        ema50_1w = ema50_1w_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Regime filter: only trade in direction of 1w trend
        bull_regime = price > ema50_1w
        bear_regime = price < ema50_1w
        
        if position == 0:
            # Long conditions: Alligator bullish + Elder Bull Power > 0 + volume spike + bull regime
            if (jaw_val < teeth_val < lips_val and  # Alligator bullish (jaw < teeth < lips)
                bull_power_val > 0 and 
                vol > 2.0 * vol_ma and
                bull_regime):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short conditions: Alligator bearish + Elder Bear Power < 0 + volume spike + bear regime
            elif (jaw_val > teeth_val > lips_val and  # Alligator bearish (jaw > teeth > lips)
                  bear_power_val < 0 and 
                  vol > 2.0 * vol_ma and
                  bear_regime):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Alligator reverses (jaw > teeth)
            if jaw_val > teeth_val:
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
            
            # Exit 1: Alligator reverses (jaw < teeth)
            if jaw_val < teeth_val:
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

name = "12h_Alligator_ElderRay_VolumeSpike_ATRStop_1wRegime"
timeframe = "12h"
leverage = 1.0