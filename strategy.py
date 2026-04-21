#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1w ADX trend filter
# Long when price breaks above Donchian(20) high, 1d volume > 2x 20-day avg, and 1w ADX > 25
# Short when price breaks below Donchian(20) low, 1d volume > 2x 20-day avg, and 1w ADX > 25
# Exit when price crosses opposite Donchian band or volume drops below average
# Donchian provides clear breakout levels, volume confirms conviction, ADX filters trending markets
# Target: 15-25 trades/year by requiring volume spike + strong trend (ADX > 25)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align all indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close[i]
        dch_high = donchian_high_aligned[i]
        dch_low = donchian_low_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        adx_val = adx_aligned[i]
        
        # Get current 1d volume (12 bars per day for 12h timeframe)
        vol_idx = i // 12
        if vol_idx >= len(df_1d):
            vol_idx = len(df_1d) - 1
        volume = df_1d['volume'].iloc[vol_idx] if vol_idx >= 0 else df_1d['volume'].iloc[0]
        
        # Volume confirmation: current 1d volume > 2x 20-day average
        volume_confirm = volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price > Donchian high, volume confirmation, ADX > 25
            if price > dch_high and volume_confirm and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian low, volume confirmation, ADX > 25
            elif price < dch_low and volume_confirm and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below Donchian low or volume drops below average
                if price < dch_low or volume < vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above Donchian high or volume drops below average
                if price > dch_high or volume < vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_1wADX25"
timeframe = "12h"
leverage = 1.0