#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold recovery) AND 1d ADX > 25 (trending) AND 1d volume spike.
# Short when Williams %R crosses below -80 (overbought breakdown) AND 1d ADX > 25 AND 1d volume spike.
# Williams %R identifies momentum reversals, ADX filters for trending markets to avoid whipsaws,
# volume confirms institutional interest. Designed for 6h timeframe to balance signal frequency and noise.
# Works in bull markets (captures oversold bounces) and bear markets (captures overbought breakdowns).
name = "6h_WilliamsR_ADX_Volume"
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
    
    # Load 1d data for ADX and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # 1d volume spike: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = np.where(vol_ema_20 > 0, df_1d['volume'].values / vol_ema_20, 1.0) > 2.0
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 6h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) > 0, williams_r, -50.0)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R cross signals
        williams_r_prev = williams_r[i-1] if i > 0 else -50
        wr_cross_above_20 = (williams_r_prev <= -20) and (williams_r[i] > -20)
        wr_cross_below_80 = (williams_r_prev >= -80) and (williams_r[i] < -80)
        
        if position == 0:
            # Long condition: WR crosses above -20, ADX > 25 (trending), volume spike
            long_condition = wr_cross_above_20 and (adx_aligned[i] > 25) and vol_spike_1d_aligned[i]
            # Short condition: WR crosses below -80, ADX > 25 (trending), volume spike
            short_condition = wr_cross_below_80 and (adx_aligned[i] > 25) and vol_spike_1d_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: WR crosses below -50 (momentum loss) or ADX weakens (< 20)
            if (williams_r[i] < -50) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: WR crosses above -50 (momentum loss) or ADX weakens (< 20)
            if (williams_r[i] > -50) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals