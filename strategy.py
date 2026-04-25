#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d ADX Regime + Volume Spike
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength relative to trend.
In strong trends (ADX > 25 on 1d), we take breakout trades in trend direction: long when Bull Power > 0 + breakout above prior high,
short when Bear Power > 0 + breakdown below prior low. Volume spike (>2.0x 20-bar MA) confirms momentum.
Uses discrete sizing (0.25) to limit fee drag. Target: 50-150 total trades over 4 years on 6h timeframe.
Works in bull markets via upside breakouts in uptrends and bear markets via downside breakdowns in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter and EMA13 for Elder Ray (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX and EMA
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = pd.Series(df_1d['close'])
    ema_13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = WilderSmooth(tr, 14)
    plus_di_1d = WilderSmooth(plus_dm, 14) / atr_1d * 100
    minus_di_1d = WilderSmooth(minus_dm, 14) / atr_1d * 100
    dx_1d = np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d) * 100
    adx_1d = WilderSmooth(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13_1d_aligned
    bear_power = ema_13_1d_aligned - low
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate prior period high/low for breakout (using 1 period ago)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ADX, EMA13, volume MA
    start_idx = max(50, 20)  # 50 for ADX (14+14+14+8 buffer), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(prior_high[i]) or
            np.isnan(prior_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        adx_val = adx_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_ma = vol_ma_20[i]
        ph = prior_high[i]
        pl = prior_low[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Elder Ray signals
        bull_strong = bull_val > 0  # Bull Power positive
        bear_strong = bear_val > 0  # Bear Power positive
        
        if position == 0:
            # Long: strong trend + Bull Power positive + break above prior high + volume confirmation
            long_signal = strong_trend and bull_strong and (curr_high > ph) and volume_confirm
            # Short: strong trend + Bear Power positive + break below prior low + volume confirmation
            short_signal = strong_trend and bear_strong and (curr_low < pl) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR price drops below prior low
            if (bull_val <= 0) or (curr_low < pl):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns negative OR price rises above prior high
            if (bear_val <= 0) or (curr_high > ph):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0