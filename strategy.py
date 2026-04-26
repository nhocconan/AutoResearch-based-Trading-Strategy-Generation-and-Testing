#!/usr/bin/env python3
"""
1d_Camarilla_H4_H5_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: 1d Camarilla H4/H5 breakout with 1w EMA50 trend filter and volume spike confirmation.
Trades only in direction of weekly trend to avoid counter-trend whipsaws in bear markets.
Volume spike ensures institutional participation. Designed for low trade frequency (10-30/year)
to minimize fee drag while capturing strong trending moves. Works in both bull and bear
markets by aligning with higher timeframe trend and requiring volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate ADX(14) on 1w for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # +DM and -DM
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
            for i in range(period, len(data)):
                if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilder_smooth(dx, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.0*(high-low), H5 = close + 1.125*(high-low)
    # L4 = close - 1.0*(high-low), L5 = close - 1.125*(high-low)
    camarilla_h4 = close_1w + 1.0 * (high_1w - low_1w)
    camarilla_h5 = close_1w + 1.125 * (high_1w - low_1w)
    camarilla_l4 = close_1w - 1.0 * (high_1w - low_1w)
    camarilla_l5 = close_1w - 1.125 * (high_1w - low_1w)
    
    # Align Camarilla levels to 1d timeframe (use previous week's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h5)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l5)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA/ADX, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h5_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_h4_aligned[i]
        breakout_strong_long = close[i] > camarilla_h5_aligned[i]
        breakout_short = close[i] < camarilla_l4_aligned[i]
        breakout_strong_short = close[i] < camarilla_l5_aligned[i]
        
        if position == 0:
            # Long: breakout above H4 AND close > 1w EMA50 AND volume spike AND ADX > 25
            if breakout_long and close[i] > ema50_1w_aligned[i] and volume_spike[i] and adx_1w_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: breakout below L4 AND close < 1w EMA50 AND volume spike AND ADX > 25
            elif breakout_short and close[i] < ema50_1w_aligned[i] and volume_spike[i] and adx_1w_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: breakout below L4 (reversal) OR weekly trend changes (close < EMA50)
            if breakout_short or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout above H4 (reversal) OR weekly trend changes (close > EMA50)
            if breakout_long or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H4_H5_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0