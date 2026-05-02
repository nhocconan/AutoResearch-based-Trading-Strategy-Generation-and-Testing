#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX regime filter and volume confirmation
# Donchian channels provide clear breakout levels that work in both trending and ranging markets
# 1d ADX > 25 filters for trending regimes to avoid whipsaws in choppy markets
# Volume confirmation at 1.8x average ensures breakouts have conviction
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Discrete sizing 0.25 to balance profit potential and fee drag

name = "12h_Donchian20_Breakout_1dADX25_Volume"
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
    
    # Calculate 1d Donchian(20) channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels: 20-period high/low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d ADX(14) for regime filter
    # Calculate +DI, -DI, and DX
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - low_1d[:-1])
    tr3 = np.abs(low_1d[1:] - high_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothing (Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize arrays
    tr_smooth = np.zeros_like(high_1d)
    plus_dm_smooth = np.zeros_like(high_1d)
    minus_dm_smooth = np.zeros_like(high_1d)
    
    # First values (simple average)
    tr_smooth[period] = np.sum(tr[:period])
    plus_dm_smooth[period] = np.sum(plus_dm[:period])
    minus_dm_smooth[period] = np.sum(minus_dm[:period])
    
    # Wilder's smoothing
    for i in range(period + 1, len(high_1d)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i-1]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i-1]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i-1]
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = np.zeros_like(high_1d)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    # ADX is smoothed DX
    adx = np.zeros_like(high_1d)
    adx[2*period-1] = np.sum(dx[period:2*period])
    for i in range(2*period, len(high_1d)):
        adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i-1]
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: Price breaks above Donchian high AND ADX > 25 AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: Price breaks below Donchian low AND ADX > 25 AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian low (breakdown) OR ADX < 20 (trend weakening)
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian high (breakout) OR ADX < 20 (trend weakening)
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals