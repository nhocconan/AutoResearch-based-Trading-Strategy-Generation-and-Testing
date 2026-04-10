#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-bar average AND 1d ADX > 25
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-bar average AND 1d ADX > 25
# - Exit when price crosses Donchian(10) midpoint OR opposite breakout occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25-35 trades/year (100-140 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong momentum moves
# - Volume confirmation ensures institutional participation
# - ADX > 25 filter ensures we only trade in trending markets (avoids chop)
# - Works in both bull (long breakouts) and bear (short breakdowns) markets

name = "4h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels on 4h data
    donchian_len = 20
    highest_high = pd.Series(prices['high'].values).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lowest_low = pd.Series(prices['low'].values).rolling(window=donchian_len, min_periods=donchian_len).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_20_avg = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (1.5 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = np.concatenate([[0], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[0], low_1d[:-1] - low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate Directional Indicators
    plus_di_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian(20) high with volume spike and ADX > 25
            if (prices['close'].iloc[i] > highest_high[i] and 
                vol_spike_1d_aligned[i] and 
                adx_1d_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian(20) low with volume spike and ADX > 25
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  vol_spike_1d_aligned[i] and 
                  adx_1d_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian(10) midpoint (mean reversion)
            # 2. Opposite breakout occurs
            donchian_len_exit = 10
            highest_high_exit = pd.Series(prices['high'].values).rolling(window=donchian_len_exit, min_periods=donchian_len_exit).max().values[i]
            lowest_low_exit = pd.Series(prices['low'].values).rolling(window=donchian_len_exit, min_periods=donchian_len_exit).min().values[i]
            donchian_mid_exit = (highest_high_exit + lowest_low_exit) / 2.0
            
            if position == 1:
                if prices['close'].iloc[i] < donchian_mid_exit or prices['close'].iloc[i] < lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if prices['close'].iloc[i] > donchian_mid_exit or prices['close'].iloc[i] > highest_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals