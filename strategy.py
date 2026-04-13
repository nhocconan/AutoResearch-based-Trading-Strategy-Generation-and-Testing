#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 1d HTF - Williams %R mean reversion with volume confirmation
    # Williams %R identifies overbought/oversold conditions; mean reversion works in ranging markets
    # Volume confirmation filters out low-volatility false signals
    # Designed for BTC/ETH ranging/mean-reverting behavior in 2025 bear market
    # Target: 50-150 trades over 4 years (12-37/year) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low + 1e-10)) * -100
    
    # Calculate 6h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)  # align volume to 1d close timing
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_confirmed = volume[i] > 1.2 * vol_avg_20_aligned[i]
        
        # Williams %R mean reversion conditions
        # Oversold: Williams %R < -80 → potential long
        # Overbought: Williams %R > -20 → potential short
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        
        # Entry conditions
        enter_long = williams_oversold and volume_confirmed
        enter_short = williams_overbought and volume_confirmed
        
        # Exit conditions: Williams %R returns to midpoint (-50)
        exit_long = position == 1 and williams_r_aligned[i] >= -50
        exit_short = position == -1 and williams_r_aligned[i] <= -50
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williamsr_meanrev_volume_v1"
timeframe = "6h"
leverage = 1.0