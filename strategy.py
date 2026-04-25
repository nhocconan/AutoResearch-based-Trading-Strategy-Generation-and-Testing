#!/usr/bin/env python3
"""
6h ADX + Williams Alligator Confluence + Volume Spike
Hypothesis: In both bull and bear markets, strong trends are confirmed when ADX > 25
and price is aligned with the Williams Alligator (jaw, teeth, lips). Adding volume
spike confirmation filters false breakouts. The Alligator provides dynamic support/resistance
while ADX ensures we only trade in trending conditions. Designed for 6h timeframe targeting
12-37 trades/year with discrete position sizing (0.0, ±0.25) to minimize fee churn.
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
    
    # Get 1d data for Williams Alligator and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA smoothed 8 periods ahead
    # Teeth: 8-period SMMA smoothed 5 periods ahead
    # Lips: 5-period SMMA smoothed 3 periods ahead
    # Using SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate ADX on 1d (needs high, low, close)
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan, dtype=float)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First TR is undefined
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            if len(data) < period:
                return np.full_like(data, np.nan, dtype=float)
            result = np.full_like(data, np.nan, dtype=float)
            # First value is sum of first 'period' values
            result[period-1] = np.nansum(data[:period])
            # Subsequent values: SMMA = Prev SMMA - (Prev SMMA / period) + Current Value
            for i in range(period, len(data)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                else:
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        atr = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        # ADX is Wilder's smoothing of DX
        adx = wilder_smooth(dx, period)
        
        return adx
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator and ADX
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: check if price is above/below all lines in proper order
        # Bullish alignment: Lips > Teeth > Jaw and price > Lips
        # Bearish alignment: Lips < Teeth < Jaw and price < Lips
        bullish_align = (lips_val > teeth_val > jaw_val) and (curr_close > lips_val)
        bearish_align = (lips_val < teeth_val < jaw_val) and (curr_close < lips_val)
        
        if position == 0:
            # Look for entry signals
            # Long: ADX > 25 (trending) AND bullish Alligator alignment AND volume spike
            long_entry = (adx_val > 25) and bullish_align and vol_spike
            # Short: ADX > 25 (trending) AND bearish Alligator alignment AND volume spike
            short_entry = (adx_val > 25) and bearish_align and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: ADX < 20 (trend weakening) OR price crosses below Teeth (Alligator signals stop)
            if (adx_val < 20) or (curr_close < teeth_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: ADX < 20 (trend weakening) OR price crosses above Teeth (Alligator signals stop)
            if (adx_val < 20) or (curr_close > teeth_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_Confluence_VolumeSpike"
timeframe = "6h"
leverage = 1.0