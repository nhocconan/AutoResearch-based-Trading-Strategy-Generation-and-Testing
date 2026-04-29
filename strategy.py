#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions (> -20 = overbought, < -80 = oversold)
# ADX > 25 filters for trending markets to avoid false reversals in ranging conditions
# Volume spike (> 2.0x 20-period MA) confirms institutional participation at extremes
# Works in bull/bear: ADX ensures we only trade reversals in established trends
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Novelty: Williams %R + ADX combination avoids whipsaws in ranging markets while capturing trend exhaustion

name = "6h_WilliamsR_Extreme_1dADX_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period) on 6h timeframe
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(df_1d['high']).diff()
    dm_minus = -pd.Series(df_1d['low']).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed TR and DM (14-period)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 20)  # warmup for Williams %R (28) and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_adx = adx_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter (ADX > 25)
            if curr_volume_confirm and curr_adx > 25:
                # Bullish entry: Williams %R crosses above -80 from below (oversold reversal)
                if i > 0 and williams_r[i-1] <= -80 and curr_wr > -80:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 from above (overbought reversal)
                elif i > 0 and williams_r[i-1] >= -20 and curr_wr < -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R crosses below -50 (momentum weakening)
            if i > 0 and williams_r[i-1] > -50 and curr_wr <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R crosses above -50 (momentum weakening)
            if i > 0 and williams_r[i-1] < -50 and curr_wr >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals