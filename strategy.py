#!/usr/bin/env python3
# Hypothesis: 12h Williams %R mean reversion with 1d ADX25 regime filter and volume confirmation.
# Long when Williams %R < -80 (oversold) in ranging market (ADX < 25) with volume > 1.5x average.
# Short when Williams %R > -20 (overbought) in ranging market (ADX < 25) with volume > 1.5x average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Williams %R identifies exhaustion points in ranging markets. ADX filter avoids trending whipsaws.
# Volume confirmation ensures participation. Works in bull markets via oversold bounces and in bear markets via overbought reversals.

name = "12h_WilliamsR_MeanReversion_1dADX25_VolumeConfirm_v1"
timeframe = "12h"
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
    
    # Calculate Williams %R (14-period)
    lookback_wr = 14
    if n < lookback_wr + 1:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1d data for ADX25 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d data
    adx_period = 14
    if len(high_1d) < adx_period + 1:
        adx_1d = np.full(len(high_1d), np.nan)
    else:
        # True Range
        tr1 = pd.Series(high_1d).diff().abs()
        tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
        tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
        
        # Directional Movement
        up_move = pd.Series(high_1d).diff()
        down_move = -pd.Series(low_1d).diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_1d = pd.Series(dx).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
    
    # Align 1d ADX to 12h timeframe (wait for 1d bar to close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback_wr + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) in ranging market (ADX < 25) with volume spike
            if (williams_r[i] < -80 and 
                adx_1d_aligned[i] < 25 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) in ranging market (ADX < 25) with volume spike
            elif (williams_r[i] > -20 and 
                  adx_1d_aligned[i] < 25 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (recovering from oversold)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (declining from overbought)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals