#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Donchian breakout, volume confirmation, and ADX trend filter
# - 1d Donchian channel (20) defines breakout levels
# - 12h volume > 1.3x 20-period average for confirmation
# - 1d ADX > 25 ensures trending market
# - Long when price breaks above upper band, short when breaks below lower band
# - Exit when price returns to middle of Donchian channel or ADX < 20
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-30 trades/year to avoid excessive fee drag

name = "12h_Donchian_20_Volume_ADX_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian, ADX, and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d ADX (14-period)
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d, prepend=low_1d[0])
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = np.maximum(
        np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - high_1d[:-1])),
        np.abs(low_1d[1:] - low_1d[:-1])
    )
    tr = np.insert(tr, 0, 0)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 12h volume for confirmation
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (00:00-23:00 UTC - trade all hours for 12h)
    # For 12h timeframe, we can trade all hours as each bar represents half a day
    # No session filter needed for 12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: 12h volume > 1.3x 12h average OR 1d volume > 1.3x 1d average
        volume_confirmation = (volume[i] > 1.3 * vol_ma_12h[i]) or \
                             (df_1d['volume'].iloc[i // (24//12)] > 1.3 * vol_ma_1d_aligned[i]) if i // (24//12) < len(df_1d) else False
        
        # ADX filter: trending market (ADX > 25)
        adx_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Look for long entry: price breaks above upper Donchian band + volume + ADX
            if close[i] > donchian_high_aligned[i] and volume_confirmation and adx_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below lower Donchian band + volume + ADX
            elif close[i] < donchian_low_aligned[i] and volume_confirmation and adx_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to middle or ADX weakens
            if close[i] < donchian_mid_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to middle or ADX weakens
            if close[i] > donchian_mid_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals