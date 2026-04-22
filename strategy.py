#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d ADX trend filter and volume confirmation.
# Williams Alligator uses three SMAs (Jaw: 13, Teeth: 8, Lips: 5) to identify trends.
# When Lips > Teeth > Jaw = bullish alignment; Lips < Teeth < Jaw = bearish.
# Combined with 1d ADX > 25 for strong trend and volume > 1.5x 20-period average.
# Designed for low trade frequency (~15-30/year) to minimize fee decay.
# Works in both bull and bear markets by requiring strong trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX on 1d
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM
    def smooth_series(x, period):
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(x[1:period])
        # Wilder smoothing
        for i in range(period, len(x)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr = smooth_series(tr, 14)
    plus_di = 100 * smooth_series(plus_dm, 14) / atr
    minus_di = 100 * smooth_series(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_series(dx, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    close = prices['close'].values
    
    def smma(series, period):
        result = np.full_like(series, np.nan)
        if len(series) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(series[:period])
        # SMMA: (prev*(period-1) + current) / period
        for i in range(period, len(series)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 20-period average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) + strong trend + volume
            if lips_val > teeth_val and teeth_val > jaw_val and strong_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) + strong trend + volume
            elif lips_val < teeth_val and teeth_val < jaw_val and strong_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator lines cross or trend weakens
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Lips < Teeth (bullish alignment breaks) or ADX < 20
                if lips_val < teeth_val or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Lips > Teeth (bearish alignment breaks) or ADX < 20
                if lips_val > teeth_val or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dADX25_Volume"
timeframe = "12h"
leverage = 1.0