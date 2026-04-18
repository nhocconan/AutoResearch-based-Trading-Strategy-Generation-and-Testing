#!/usr/bin/env python3
"""
6h ADX + Volume Spike with Daily EMA Trend Filter
Hypothesis: ADX > 25 indicates strong trend, combined with volume spike confirms momentum.
Daily EMA50 filters direction to avoid counter-trend trades. Designed for 12-37 trades/year
on 6h timeframe with low trade frequency to minimize fee drift. Works in bull/bear by
requiring volume spike and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Daily EMA50 for trend filter
    ema_50_d = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_d, ema_50_d)
    
    # ADX calculation on 6h data
    # +DM, -DM, TR
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    
    # Pad arrays to original length
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    tr = np.concatenate([[0.0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # simple average for first value
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    plus_di = 100 * wilders_smoothing(plus_dm, period) / wilders_smoothing(tr, period)
    minus_di = 100 * wilders_smoothing(minus_dm, period) / wilders_smoothing(tr, period)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(adx[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx[i]
        ema_trend = ema_50_aligned[i]
        
        if position == 0:
            # Long: ADX > 25, volume spike, price above EMA (uptrend)
            if adx_val > 25 and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25, volume spike, price below EMA (downtrend)
            elif adx_val > 25 and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: ADX drops below 20 (trend weakening) or price crosses below EMA
            if adx_val < 20 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: ADX drops below 20 (trend weakening) or price crosses above EMA
            if adx_val < 20 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_VolumeSpike_EMA50_Trend"
timeframe = "6h"
leverage = 1.0