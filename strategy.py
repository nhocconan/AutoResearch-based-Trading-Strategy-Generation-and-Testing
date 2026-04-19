#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w ADX trend filter
# - Long when price breaks above Donchian(20) high + 1d volume > 1.5x average + 1w ADX > 25
# - Short when price breaks below Donchian(20) low + 1d volume > 1.5x average + 1w ADX > 25
# - Exit when price crosses Donchian midpoint or ADX < 20 (trend weakening)
# - Designed to capture strong trends with volume confirmation in both bull and bear markets
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "4h_Donchian20_1dVolume_1wADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w ADX calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / tr_smooth
    minus_di = 100 * wilder_smooth(minus_dm, 14) / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    adx_1w = adx
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 6.0)
        
        if position == 0:
            # Look for long entry: breakout above Donchian high + volume + strong trend (ADX > 25)
            if close[i] > donchian_high[i] and volume_filter and adx_1w_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Look for short entry: breakout below Donchian low + volume + strong trend (ADX > 25)
            elif close[i] < donchian_low[i] and volume_filter and adx_1w_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below Donchian midpoint or trend weakens (ADX < 20)
            if close[i] < donchian_mid[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above Donchian midpoint or trend weakens (ADX < 20)
            if close[i] > donchian_mid[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals