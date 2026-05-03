#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX25 trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions. Extreme readings (<-90 or >-10)
# combined with 1d ADX > 25 (strong trend) and volume spike capture momentum exhaustion
# reversals in both bull and bear markets. Designed for 12-30 trades/year on 6h to
# minimize fee drag while maintaining edge via confluence of momentum, trend, and volume.

name = "6h_WilliamsR_Extreme_1dADX25_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR and DM
    tr_ma = pd.Series(tr).ewm(span=25, adjust=False, min_periods=25).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(span=25, adjust=False, min_periods=25).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=25, adjust=False, min_periods=25).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R (14-period) on 6h data
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after sufficient warmup for Williams %R
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R calculation using data up to current bar
        lookback = min(14, i+1)
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        
        if highest_high == lowest_low:
            williams_r = -50  # avoid division by zero
        else:
            williams_r = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        
        # Volume spike confirmation (20-period EMA)
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (2.0 * vol_ema_20)
        
        # Extreme Williams %R conditions with trend filter and volume
        williams_oversold = williams_r < -90  # extreme oversold
        williams_overbought = williams_r > -10  # extreme overbought
        strong_trend = adx_1d_aligned[i] > 25  # strong 1d trend
        
        if position == 0:
            # Long: extreme oversold in strong trend with volume spike
            if williams_oversold and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: extreme overbought in strong trend with volume spike
            elif williams_overbought and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 or trend weakens
            if williams_r > -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 or trend weakens
            if williams_r < -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals