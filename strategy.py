#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX(14) trend filter
# - Long when price breaks above Donchian(20) high + ADX(14) > 25 + 1d volume > 1.5x 20-period volume SMA
# - Short when price breaks below Donchian(20) low + same ADX and volume conditions
# - Exit: opposite Donchian breakout (long exits on lower band break, short exits on upper band break)
# - Position sizing: 0.25 discrete level to balance risk and reward
# - Donchian channels provide clear breakout levels in both trending and ranging markets
# - ADX filter ensures we only trade when there's sufficient trend strength
# - Volume confirmation adds conviction to breakouts
# - Target: 20-50 trades/year on 4h timeframe to minimize fee drag

name = "4h_1d_donchian_volume_adx_v1"
timeframe = "4h"
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
    
    # Calculate 4h Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate 4h ADX for trend filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_period = 14
    atr = pd.Series(tr1).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, np.finfo(float).eps, atr)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values / atr_safe
    minus_di = 100 * pd.Series(minus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values / atr_safe
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero in DX calculation
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        
        # Volume confirmation: current 1d volume > 1.5x 20-period SMA (volume spike)
        vol_confirm = volume_1d_current[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Donchian breakout signals
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Exit conditions: opposite band break
        exit_long = close[i] < lowest_low[i]
        exit_short = close[i] > highest_high[i]
        
        if position == 0:  # Flat - look for entry
            if long_breakout and trending and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif short_breakout and trending and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals