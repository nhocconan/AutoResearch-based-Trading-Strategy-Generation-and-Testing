#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with 12-hour volume confirmation and 1-day ADX trend filter
# Williams Alligator (13,8,5 SMAs) identifies trend direction and strength
# In strong trends (ADX > 25), trade Alligator crossovers with volume confirmation
# Volume spike filters low-conviction moves; ADX filter avoids whipsaws in ranging markets
# Target: 20-35 trades/year to minimize fee decay while capturing strong trends

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Williams Alligator (13,8,5 SMAs) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Jaw (13-period SMA)
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMA)
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMA)
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # === 12h Volume Spike (vs 20-period average) ===
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # === 1-day ADX (14-period) for trend strength ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_smooth / atr_1d
    minus_di_1d = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 12h volume (avoid calling get_htf_data in loop)
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        
        # Volume spike: current 12h volume > 1.5x 20-period average
        vol_spike = volume_12h_aligned[i] > vol_ma_20_12h_aligned[i] * 1.5
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        # Alligator signals
        # Bullish: Lips > Teeth > Jaw (green alignment)
        bullish_aligned = (lips_12h_aligned[i] > teeth_12h_aligned[i]) and (teeth_12h_aligned[i] > jaw_12h_aligned[i])
        # Bearish: Lips < Teeth < Jaw (red alignment)
        bearish_aligned = (lips_12h_aligned[i] < teeth_12h_aligned[i]) and (teeth_12h_aligned[i] < jaw_12h_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            if strong_trend and vol_spike:
                if bullish_aligned:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif bearish_aligned:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic: exit when Alligator alignment breaks or trend weakens
        elif position == 1:
            # Exit long if bearish alignment forms or trend weakens
            if bearish_aligned or not strong_trend:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if bullish alignment forms or trend weakens
            if bullish_aligned or not strong_trend:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ADXTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0