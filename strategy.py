#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Bollinger Bands breakout with 1d ADX25 trend filter and volume spike
    # Bollinger Bands provide dynamic support/resistance; breakouts indicate momentum shifts
    # ADX25 on 1d filters for strong trending markets (avoids ranging conditions)
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in bull/bear: breaks through volatility bands with trend and volume confirmation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2
    ma20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = ma20 + bb_std * std20
    lower_band = ma20 - bb_std * std20
    
    # Load 1d data for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14) on 1d
    adx_period = 14
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    atr = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ma20[i]) or 
            np.isnan(std20[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Bollinger Band with ADX > 25 and volume spike
            if close[i] > upper_band[i] and adx_aligned[i] > 25 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Bollinger Band with ADX > 25 and volume spike
            elif close[i] < lower_band[i] and adx_aligned[i] > 25 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to middle Bollinger Band (mean reversion within trend)
            if position == 1:
                if close[i] < ma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Bollinger_Bands_Breakout_ADX25_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0