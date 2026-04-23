#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 Breakout with 1d ADX Trend Filter and Volume Spike
- Camarilla R4/S4 levels (stronger than R3/S3) from prior 1d provide significant support/resistance
- 1d ADX > 25 ensures alignment with strong daily trend for multi-timeframe confirmation
- Volume > 2.0x 20-period average confirms breakout momentum with very strict filtering
- Designed for 4h timeframe targeting 20-40 trades/year (80-160 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with strong trend, in bear markets via fade of overextended moves at strong levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ADX trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R4, S4 levels: R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 1d ADX(14) for trend filter
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Calculate DI and ADX
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align ADX to 4h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: > 2.0x 20-period average (very strict)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # ADX needs ~34 bars, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout signals with strong trend filter and volume spike
        # Long: price breaks above Camarilla R4 + strong uptrend (ADX>25) + volume spike
        # Short: price breaks below Camarilla S4 + strong downtrend (ADX>25) + volume spike
        long_signal = (close[i] > camarilla_r4_aligned[i] and 
                      adx_aligned[i] > 25 and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < camarilla_s4_aligned[i] and 
                       adx_aligned[i] > 25 and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend weakening (ADX<20) or opposite Camarilla level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend weakening or price breaks below Camarilla S4
                if (adx_aligned[i] < 20 or 
                    close[i] < camarilla_s4_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend weakening or price breaks above Camarilla R4
                if (adx_aligned[i] < 20 or 
                    close[i] > camarilla_r4_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R4S4_Breakout_1dADX_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0