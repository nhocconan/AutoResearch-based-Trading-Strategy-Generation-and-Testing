#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h ADX trend filter and volume spike confirmation.
# Donchian(20) breakouts capture momentum, ADX(14) > 25 ensures strong trend direction,
# volume > 2x average confirms institutional participation. Works in both bull and bear markets.
# Target: 20-50 trades/year with disciplined entries to minimize fee drag.

name = "4h_Donchian_Breakout_12hADX25_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        plus_di = np.full_like(tr, np.nan)
        minus_di = np.full_like(tr, np.nan)
        
        # Wilder's smoothing (first value is simple average)
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_sum = np.nansum(plus_dm[1:period+1])
        minus_dm_sum = np.nansum(minus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
        
        # Avoid division by zero
        atr[atr == 0] = 1e-10
        
        plus_di[period:] = (plus_dm_sum / atr[period:]) * 100
        minus_di[period:] = (minus_dm_sum / atr[period:]) * 100
        
        # DX and ADX
        dx = np.full_like(tr, np.nan)
        dx[period:] = np.abs((plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:] + 1e-10)) * 100
        
        adx = np.full_like(tr, np.nan)
        adx[2*period] = np.nanmean(dx[period:2*period+1])
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_20[i] = np.mean(volume_12h[i-20:i])
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Align all indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 2.0x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 12h bar's volume
            idx_12h = 0
            while idx_12h < len(df_12h) and df_12h.iloc[idx_12h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_12h += 1
            idx_12h -= 1  # last completed 12h bar
            
            if idx_12h >= 0:
                vol_12h_current = df_12h.iloc[idx_12h]['volume']
                vol_filter = vol_12h_current > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout + ADX > 25 + volume spike
            # Long when price breaks above upper Donchian band in uptrend
            long_condition = (close[i] > donchian_high[i]) and \
                             (adx_aligned[i] > 25) and vol_filter
            # Short when price breaks below lower Donchian band in downtrend
            short_condition = (close[i] < donchian_low[i]) and \
                              (adx_aligned[i] > 25) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian band or ADX < 20
            if (close[i] < donchian_low[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above upper Donchian band or ADX < 20
            if (close[i] > donchian_high[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals