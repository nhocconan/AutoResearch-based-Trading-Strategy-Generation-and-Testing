#!/usr/bin/env python3
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
    
    # Get daily data for Williams Alligator components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = low_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three smoothed moving averages
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    # We'll use the standard approach: SMMA with period, then shift
    
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    # Calculate SMMA on daily close
    smma5 = smma(close_1d, 5)
    smma8 = smma(close_1d, 8)
    smma13 = smma(close_1d, 13)
    
    # Shift as per Alligator specification
    lips = np.roll(smma5, 3)    # 5-period, shifted 3 bars
    teeth = np.roll(smma8, 5)   # 8-period, shifted 5 bars
    jaw = np.roll(smma13, 8)    # 13-period, shifted 8 bars
    
    # Handle initial NaN values from rolling
    lips[:3] = np.nan
    teeth[:5] = np.nan
    jaw[:8] = np.nan
    
    # Align to 6h timeframe
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    
    # ADX calculation on daily data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = np.nan
        tr2[0] = np.nan
        tr3[0] = np.nan
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = np.nan
        down_move[0] = np.nan
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        def wilders_smooth(data, period):
            result = np.full_like(data, np.nan, dtype=float)
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period])
                for i in range(period, len(data)):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        tr_sum = wilders_smooth(tr, period)
        plus_dm_sum = wilders_smooth(plus_dm, period)
        minus_dm_sum = wilders_smooth(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_sum != 0, 100 * plus_dm_sum / tr_sum, 0)
        minus_di = np.where(tr_sum != 0, 100 * minus_dm_sum / tr_sum, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation on 6h: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lips_6h[i]) or 
            np.isnan(teeth_6h[i]) or 
            np.isnan(jaw_6h[i]) or
            np.isnan(adx_6h[i]) or
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = (lips_6h[i] > teeth_6h[i]) and (teeth_6h[i] > jaw_6h[i])
        bearish_alignment = (lips_6h[i] < teeth_6h[i]) and (teeth_6h[i] < jaw_6h[i])
        
        # ADX filter: only trade when trend is strong enough
        strong_trend = adx_6h[i] > 25
        
        if position == 0:
            # Long: Bullish Alligator alignment + strong trend + volume
            if bullish_alignment and strong_trend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + strong trend + volume
            elif bearish_alignment and strong_trend and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks down or trend weakens
            if not bullish_alignment or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks down or trend weakens
            if not bearish_alignment or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ADX_Volume"
timeframe = "6h"
leverage = 1.0