#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# - Long when price breaks above 4h Donchian upper channel AND ATR(14) > ATR(50) (expanding volatility)
# - Short when price breaks below 4h Donchian lower channel AND ATR(14) > ATR(50) (expanding volatility)
# - Volume confirmation: 4h volume > 1.3x 20-period volume SMA
# - Exit: opposite Donchian breakout or volatility contraction (ATR(14) < ATR(50))
# - Position sizing: 0.25 discrete level
# - Target: 20-50 trades/year on 4h timeframe to stay within fee drag limits
# - Works in both bull and breakout markets by capturing volatility expansion moves

name = "4h_1d_atr_vol_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Calculate 1d ATR for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align ATR to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Calculate 4h volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Volatility regime filter: ATR(14) > ATR(50) (expanding volatility)
        vol_expanding = atr_14_aligned[i] > atr_50_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Break above previous upper channel
        breakout_down = close[i] < donchian_lower[i-1]  # Break below previous lower channel
        
        # Exit conditions: opposite breakout or volatility contraction
        exit_long = breakout_down or not vol_expanding
        exit_short = breakout_up or not vol_expanding
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_expanding and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_expanding and vol_confirm:
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