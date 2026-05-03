#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter
# Camarilla levels provide precise intraday support/resistance, volume spike confirms institutional participation,
# ADX > 25 ensures we only trade in trending markets to avoid whipsaws in ranges.
# Designed to capture strong momentum moves while filtering false breakouts.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADXTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume spike and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx) | (plus_di + minus_di == 0), 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align 1d indicators to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    camarilla_r3 = close + 1.1 * (high - low)
    camarilla_s3 = close - 1.1 * (high - low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Price breaks above R3 + volume spike + trending market
            if close[i] > camarilla_r3[i] and volume_spike_aligned[i] and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + volume spike + trending market
            elif close[i] < camarilla_s3[i] and volume_spike_aligned[i] and is_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below midpoint of daily range OR reverse signal
            midpoint = (high[i] + low[i]) / 2
            if close[i] < midpoint or (close[i] < camarilla_s3[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above midpoint of daily range OR reverse signal
            midpoint = (high[i] + low[i]) / 2
            if close[i] > midpoint or (close[i] > camarilla_r3[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals