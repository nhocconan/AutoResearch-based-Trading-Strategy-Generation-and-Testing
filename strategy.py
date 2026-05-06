#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian breakout with volume confirmation and ADX trend filter
# Daily Donchian channels (20-period high/low) identify key support/resistance levels
# Breakout above 20-day high or below 20-day low with volume > 2.0x 20-period average indicates strong momentum
# ADX(14) > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions
# Works in bull/bear markets: breakouts capture new trends, ADX filter avoids false signals in consolidation
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_DailyDonchian20_VolumeADXFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Donchian channels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: >2.0x 20-period average (high threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # ADX trend filter on 12h timeframe
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = np.full_like(tr, np.nan)
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    
    # Wilder's smoothing: first value is simple average, then recursive
    period = 14
    if len(tr) >= period:
        # Initial values
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_sum = np.nansum(plus_dm[1:period+1])
        minus_dm_sum = np.nansum(minus_dm[1:period+1])
        
        # Smooth subsequent values
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            plus_dm_smoothed = (plus_di[i-1] * (period - 1) + plus_dm[i]) / period if not np.isnan(plus_di[i-1]) else np.nan
            minus_dm_smoothed = (minus_di[i-1] * (period - 1) + minus_dm[i]) / period if not np.isnan(minus_di[i-1]) else np.nan
            
            # Avoid division by zero
            if not np.isnan(atr[i]) and atr[i] != 0:
                plus_di[i] = (plus_dm_smoothed / atr[i]) * 100 if not np.isnan(plus_dm_smoothed) else 0
                minus_di[i] = (minus_dm_smoothed / atr[i]) * 100 if not np.isnan(minus_dm_smoothed) else 0
            else:
                plus_di[i] = 0
                minus_di[i] = 0
        
        # Calculate ADX
        dx = np.full_like(tr, np.nan)
        adx = np.full_like(tr, np.nan)
        
        for i in range(period*2, len(tr)):  # Need enough data for ADX
            if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
                dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # Smooth DX to get ADX
        if len(dx) >= period*2:
            adx[period*2] = np.nansum(dx[period:period*2+1]) / period
            for i in range(period*2 + 1, len(tr)):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # ADX > 25 indicates strong trend
    adx_filter = adx > 25 if 'adx' in locals() else np.full_like(tr, False)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(volume_filter[i]) or (i < len(adx_filter) and np.isnan(adx_filter[i])) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_filter[i] if i < len(adx_filter) else False
        
        if position == 0:
            # Long breakout: price breaks above 20-day high with volume confirmation and strong trend
            if close[i] > high_20_aligned[i] and volume_filter[i] and adx_val:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 20-day low with volume confirmation and strong trend
            elif close[i] < low_20_aligned[i] and volume_filter[i] and adx_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-day low (trend reversal)
            if close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-day high (trend reversal)
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals