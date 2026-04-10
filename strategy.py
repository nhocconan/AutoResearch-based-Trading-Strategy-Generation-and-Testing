#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter and volume spike
# - Williams %R(14) identifies overbought/oversold conditions on 6h
# - ADX(14) from 1d timeframe filters regime: ADX < 25 = range (mean revert), ADX > 25 = trend (avoid)
# - Volume spike confirmation: 6h volume > 1.5x 20-period volume SMA
# - Long: Williams %R < -80 (oversold) AND ADX < 25 AND volume spike
# - Short: Williams %R > -20 (overbought) AND ADX < 25 AND volume spike
# - Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
# - Position sizing: 0.25 discrete level
# - Target: 12-35 trades/year on 6h timeframe to stay within fee drag limits
# - Works in both bull/bear markets by fading extremes only in ranging regimes

name = "6h_1d_williamsr_meanreversion_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 6h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d ADX(14) for regime filter
    # ADX calculation requires +DI and -DI
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14 * 100
    minus_di_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14 * 100
    
    # DX and ADX
    dx = np.abs(plus_di_14 - minus_di_14) / (np.abs(plus_di_14 + minus_di_14)) * 100
    dx = np.where((plus_di_14 + minus_di_14) == 0, 0, dx)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 6h volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Regime filter: ADX < 25 = ranging market (good for mean reversion)
        ranging_market = adx_14_aligned[i] < 25
        
        # Williams %R signals
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_long = williams_r[i] > -50  # Exit long when crosses above -50
        exit_short = williams_r[i] < -50  # Exit short when crosses below -50
        
        if position == 0:  # Flat - look for entry
            if oversold and ranging_market and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif overbought and ranging_market and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long or not ranging_market or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short or not ranging_market or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals