#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R mean reversion with 1d volume spike and ADX trend filter.
# Long when Williams %R < -80 (oversold), volume > 1.5x 20-period average, and ADX > 25 (trending).
# Short when Williams %R > -20 (overbought), volume > 1.5x 20-period average, and ADX > 25.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position size 0.25. Williams %R identifies extremes, volume confirms participation,
# ADX ensures we trade with momentum, not against it. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Williams %R (14-period) ===
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 13 or highest_high[i] == lowest_low[i]:
            williams_r[i] = -50.0  # neutral
        else:
            williams_r[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Williams %R signals: 1 for oversold (< -80), -1 for overbought (> -20), 0 otherwise
    williams_signal = np.zeros_like(williams_r)
    williams_signal[williams_r < -80] = 1   # oversold -> long signal
    williams_signal[williams_r > -20] = -1  # overbought -> short signal
    
    # === 1d Indicators: Volume Spike (20-period average) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = np.zeros_like(volume_1d)
    volume_spike = volume_1d > (1.5 * vol_ma_20_1d)  # boolean array
    
    # === 1d Indicators: ADX (14-period) for trend strength ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = np.zeros_like(close_1d)
    minus_di = np.zeros_like(close_1d)
    dx = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if atr[i] > 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        else:
            plus_di[i] = 0
            minus_di[i] = 0
            dx[i] = 0
    
    # ADX: smoothed DX
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_filter = adx > 25  # trending market
    
    # Align all 1d indicators to 12h timeframe
    williams_signal_aligned = align_htf_to_ltf(prices, df_1d, williams_signal)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_signal_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(adx_filter_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        williams_sig = williams_signal_aligned[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # convert back to boolean
        adx_filt = adx_filter_aligned[i] > 0.5      # convert back to boolean
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 (no longer oversold)
            if williams_sig != 1:  # not oversold anymore
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 (no longer overbought)
            if williams_sig != -1:  # not overbought anymore
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # All filters must align: signal, volume spike, and trend
            if williams_sig == 1 and vol_spike and adx_filt:
                signals[i] = 0.25
                position = 1
            elif williams_sig == -1 and vol_spike and adx_filt:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dWilliamsR_VolumeSpike_ADXFilter_V1"
timeframe = "12h"
leverage = 1.0