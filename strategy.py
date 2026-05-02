#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX(14) regime filter
# Uses 4h primary timeframe with 1d HTF for ADX trend strength filter.
# Breakouts in direction of 1d ADX > 25 (trending) with volume spike capture strong moves.
# Designed for low trade frequency (19-50/year) to minimize fee drag in 4h timeframe.
# Works in both bull and bear markets by only taking trades when ADX confirms trending regime.

name = "4h_Donchian20_VolumeSpike_ADX25_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX(14) trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d for trend strength filter
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate +DM, -DM, TR
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(close_1d[1:] - close_1d[:-1]))
    tr = np.maximum(tr, np.absolute(high_1d[1:] - close_1d[:-1]))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    tr = np.concatenate([[0.0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    tr_sum = np.zeros_like(tr)
    plus_dm_sum = np.zeros_like(plus_dm)
    minus_dm_sum = np.zeros_like(minus_dm)
    
    tr_sum[0] = tr[0]
    plus_dm_sum[0] = plus_dm[0]
    minus_dm_sum[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i]
        plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm[i]
        minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm[i]
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_sum / (tr_sum + 1e-10)
    minus_di = 100 * minus_dm_sum / (tr_sum + 1e-10)
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros_like(dx)
    adx[period-1] = np.mean(dx[:period]) if period <= len(dx) else 0
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align 1d ADX to 4h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian(20) channels on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (2.0x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(34 for ADX calc, 20 for Donchian +1 for shift, 20 for volume +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + ADX > 25 (trending) + volume spike
            if (close[i] > donchian_high[i] and adx_1d_aligned[i] > 25 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + ADX > 25 (trending) + volume spike
            elif (close[i] < donchian_low[i] and adx_1d_aligned[i] > 25 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns below Donchian low (mean reversion) or ADX < 20 (trend weakening)
            if close[i] < donchian_low[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns above Donchian high (mean reversion) or ADX < 20 (trend weakening)
            if close[i] > donchian_high[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals