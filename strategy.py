#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + ADX(14) trend filter
# - Long when price breaks above 20-period Donchian high (12h) + ADX(14) > 25 + 1d volume > 2.0x 20-period volume SMA
# - Short when price breaks below 20-period Donchian low (12h) + same ADX and volume conditions
# - Exit: price returns to 12-period Donchian midpoint (mean of high/low)
# - Position sizing: 0.30 discrete level
# - Donchian channels provide clear trend-following breakout levels
# - ADX filter ensures we only trade in trending markets, avoiding chop
# - Volume confirmation ensures breakout strength
# - Target: 12-37 trades/year on 12h timeframe to minimize fee drag

name = "12h_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate 12h Donchian channels (20-period)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 12h ADX for trend filter
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
    
    # Align Donchian levels to 12h timeframe (already in 12h, but keep for consistency)
    donchian_high_aligned = donchian_high  # Already in primary timeframe
    donchian_low_aligned = donchian_low
    donchian_mid_aligned = donchian_mid
    adx_aligned = adx
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 2.0x 20-period SMA (strong volume spike)
        vol_confirm = volume_1d_current[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Donchian breakout signals
        long_entry = (close[i] > donchian_high_aligned[i]) and trending and vol_confirm
        short_entry = (close[i] < donchian_low_aligned[i]) and trending and vol_confirm
        exit_long = close[i] < donchian_mid_aligned[i]  # Exit long when price crosses below midpoint
        exit_short = close[i] > donchian_mid_aligned[i]  # Exit short when price crosses above midpoint
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.30
            elif short_entry:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
    
    return signals