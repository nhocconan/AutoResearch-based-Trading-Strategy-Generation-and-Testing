#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d ADX trend filter and volume confirmation
# Williams %R measures overbought/oversold: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R crosses above -80 from below AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Short when %R crosses below -20 from above AND 1d ADX > 25 AND volume > 1.5x 20 EMA
# Uses 12h timeframe for lower frequency, Williams %R for momentum reversals, 1d ADX to filter choppy markets,
# volume confirmation to avoid false signals. Designed for 12-37 trades/year with discrete sizing (0.25).
# Works in bull markets via buying oversold dips in uptrends and bear markets via selling overbought rallies in downtrends.

name = "12h_WilliamsR_1dADX_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF ADX filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_trending = adx > 25  # Strong trend filter
    
    # Align 1d ADX trend to 12h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    
    # Calculate 12h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate Williams %R crossover signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]
    
    # Long signal: %R crosses above -80 from below
    long_signal = (williams_r > -80) & (williams_r_prev <= -80)
    # Short signal: %R crosses below -20 from above
    short_signal = (williams_r < -20) & (williams_r_prev >= -20)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_trending_aligned[i]) or np.isnan(long_signal[i]) or 
            np.isnan(short_signal[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 AND 1d trending AND volume spike
            if (long_signal[i] and 
                adx_trending_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 AND 1d trending AND volume spike
            elif (short_signal[i] and 
                  adx_trending_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -20 (overbought) OR 1d trend weakens
            if (williams_r[i] >= -20 or 
                adx_trending_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -80 (oversold) OR 1d trend weakens
            if (williams_r[i] <= -80 or 
                adx_trending_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals