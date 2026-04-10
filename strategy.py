#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion + 1d volume spike + ADX(14) trend filter
# - Long when Williams %R(14) < -80 (oversold) + ADX(14) > 25 + 1d volume > 2.0x 20-period volume SMA
# - Short when Williams %R(14) > -20 (overbought) + same ADX and volume conditions
# - Exit: Williams %R returns to -50 (mean reversion midpoint)
# - Position sizing: 0.30 discrete level
# - Williams %R identifies overextended moves ready for mean reversion
# - ADX filter ensures we only trade in trending markets, avoiding chop
# - Volume confirmation ensures breakout/mean reversion strength
# - Target: 12-37 trades/year on 12h timeframe to minimize fee drag

name = "12h_williamsr_volume_adx_v1"
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
    
    # Calculate 12h Williams %R (14-period)
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
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
    
    # Align Williams %R and ADX to 12h timeframe (already in 12h, but keep for consistency)
    williams_r_aligned = williams_r
    adx_aligned = adx
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 2.0x 20-period SMA (strong volume spike)
        vol_confirm = volume_1d_current[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Williams %R mean reversion signals
        long_entry = (williams_r_aligned[i] < -80) and trending and vol_confirm
        short_entry = (williams_r_aligned[i] > -20) and trending and vol_confirm
        exit_long = williams_r_aligned[i] > -50  # Exit long when Williams %R crosses above -50
        exit_short = williams_r_aligned[i] < -50  # Exit short when Williams %R crosses below -50
        
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