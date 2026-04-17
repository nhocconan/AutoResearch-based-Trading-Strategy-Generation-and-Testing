#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d volume spike and 1d ADX trend filter
# Williams %R identifies overbought/oversold conditions. Reversals occur when %R crosses back from extremes.
# Works in bull/bear by taking reversals from oversold in uptrend and overbought in downtrend.
# Position size: 0.25 for balanced risk/return.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for volume, ADX, and Williams %R ===
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX calculation (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d volume and its 20-period average
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # 1d Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or \
           np.isnan(williams_r_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 20-period average volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > volume_ma20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_1d_aligned[i] > 25
        
        # Combined filter
        filter_ok = volume_filter and trend_filter
        
        if position == 0:
            # Long when Williams %R crosses above -80 from oversold in uptrend
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                adx_1d_aligned[i] > 25 and filter_ok):
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -20 from overbought in downtrend
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  adx_1d_aligned[i] > 25 and filter_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses below -20 (overbought) or filter fails
            if (williams_r_aligned[i] < -20 or 
                not filter_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses above -80 (oversold) or filter fails
            if (williams_r_aligned[i] > -80 or 
                not filter_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dVolume_ADX_Filter"
timeframe = "12h"
leverage = 1.0