#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX regime filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d ADX > 25 (trending) AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND 1d ADX > 25 AND volume > 1.5x 20-bar avg
# - Exit when price crosses Donchian(10) midpoint (faster exit to reduce drawdown)
# - Uses 1d ADX for regime filter to avoid choppy markets where breakouts fail
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 30-50 trades/year on 4h timeframe (120-200 total over 4 years)
# - Donchian breakouts work in trending markets; ADX filter ensures we only trade when trends exist

name = "4h_1d_donchian_breakout_adx_volume_v1"
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
    
    # Pre-compute Donchian channels from 4h data (primary timeframe)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian(20) for breakout signals
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian(10) for faster exits
    donch_high_10 = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    donch_low_10 = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    donch_mid_10 = (donch_high_10 + donch_low_10) / 2
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed TR and DM
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_14 / tr_14)
    di_minus = 100 * (dm_minus_14 / tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF ADX to LTF
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for Donchian(20)
        # Skip if any required data is invalid
        if (np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or 
            np.isnan(donch_mid_10[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian(20) high AND trending regime with volume spike
            if (prices['close'].iloc[i] > donch_high_20[i] and 
                adx_aligned[i] > 25 and  # trending market
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian(20) low AND trending regime with volume spike
            elif (prices['close'].iloc[i] < donch_low_20[i] and 
                  adx_aligned[i] > 25 and  # trending market
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at Donchian(10) midpoint
            # Exit when price crosses Donchian(10) midpoint
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= donch_mid_10[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= donch_mid_10[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals