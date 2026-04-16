#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R(14) mean reversion with 1d ADX(14) regime filter and volume confirmation.
# Long when Williams %R < -80 AND 1d ADX < 20 (strong range regime) AND volume > 1.3x 24-period average.
# Short when Williams %R > -20 AND 1d ADX < 20 (strong range regime) AND volume > 1.3x 24-period average.
# Uses discrete position size 0.25. Williams %R captures overextended moves in ranging markets, 
# 1d ADX ensures we only trade when higher timeframe is strongly ranging (avoiding whipsaws),
# volume spike confirms participation. Designed to work in both bull (buy dips) and bear (sell rallies) 
# markets during strong ranging conditions. Target: 80-120 trades over 4 years (20-30/year) to balance 
# opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan)
    williams_r_values = williams_r.values
    
    # === 12h Indicators: Volume Spike (volume > 1.3x 24-period average) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    volume_spike = volume > (1.3 * vol_ma)
    volume_spike_values = volume_spike.values
    
    # Get 1d data once before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX(14) for regime filter ===
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ADX, 24 for volume MA, 14 for Williams %R)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_values[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma.values[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        williams_r_val = williams_r_values[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike_values[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R returns to oversold threshold (-50) or volume spike ends
            if williams_r_val >= -50 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R returns to overbought threshold (-50) or volume spike ends
            if williams_r_val <= -50 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 AND 1d ADX < 20 (strong range regime) AND volume spike
            if williams_r_val < -80 and adx_val < 20 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 AND 1d ADX < 20 (strong range regime) AND volume spike
            elif williams_r_val > -20 and adx_val < 20 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_WilliamsR14_1dADX20_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0