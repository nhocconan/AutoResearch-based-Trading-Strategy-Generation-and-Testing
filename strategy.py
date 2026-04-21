#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ADX trend filter
# Long when price breaks above Donchian(20) high and volume > 2x 20-period average and ADX > 25
# Short when price breaks below Donchian(20) low and volume > 2x 20-period average and ADX > 25
# Exit when price crosses Donchian(20) midpoint or ADX falls below 20
# Donchian channels provide clear breakout levels, volume confirms conviction, ADX filters trending markets
# Target: 20-35 trades/year by requiring volume spike + trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian(20) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        donch_mid_val = donchian_mid[i]
        vol_ma = vol_ma_1d_aligned[i]
        adx_val = adx_aligned[i]
        volume = df_1d['volume'].iloc[i // 6] if i >= 6 else df_1d['volume'].iloc[0]  # 6 bars per day (24h/4h)
        
        # Volume confirmation: current 1d volume > 2x 20-day average
        volume_confirm = volume > 2.0 * vol_ma if i >= 6 else df_1d['volume'].iloc[0] > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high, volume confirmation, ADX > 25
            if price > donch_high and volume_confirm and adx_val > 25:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low, volume confirmation, ADX > 25
            elif price < donch_low and volume_confirm and adx_val > 25:
                signals[i] = -0.30
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below Donchian midpoint or ADX < 20
                if price < donch_mid_val or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above Donchian midpoint or ADX < 20
                if price > donch_mid_val or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dVolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0