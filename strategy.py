#!/usr/bin/env python3
# Hypothesis: 4h Williams %R with 1d ADX trend filter and volume spike
# Long when Williams %R crosses above -20 (oversold reversal) with ADX > 25 and volume > 1.5x average
# Short when Williams %R crosses below -80 (overbought reversal) with ADX > 25 and volume > 1.5x average
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme
# Williams %R captures momentum reversals, ADX filters for trending conditions, volume confirms conviction
# Designed for 4-8 trades per month (~50-100/year) with controlled risk in all market regimes

name = "4h_WilliamsR_ADX_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d ADX (14-period) for trend filter
    # ADX requires +DI and -DI calculation
    plus_dm = np.where((df_1d['high'].diff()) > (df_1d['low'].diff().abs()), df_1d['high'].diff(), 0)
    minus_dm = np.where((df_1d['low'].diff().abs()) > (df_1d['high'].diff()), df_1d['low'].diff().abs(), 0)
    plus_dm = np.where(plus_dm < 0, 0, plus_dm)
    minus_dm = np.where(minus_dm < 0, 0, minus_dm)
    
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r_aligned[i-1]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -20 (from oversold), ADX > 25, volume spike
            if (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 and
                adx_aligned[i] > 25 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -80 (from overbought), ADX > 25, volume spike
            elif (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 and
                  adx_aligned[i] > 25 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 or crosses below -80
            if (williams_r_aligned[i] >= -50) or (williams_r_aligned[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 or crosses above -20
            if (williams_r_aligned[i] <= -50) or (williams_r_aligned[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals