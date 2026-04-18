#!/usr/bin/env python3
"""
4h_DonchianBreakout_VolumeTrend_v3
Hypothesis: Trade Donchian channel breakouts (20-period) in the direction of 1d EMA(50) trend, confirmed by volume >1.5x average. Uses ATR-based stop loss to limit drawdown. Designed for fewer trades (<50/year) to avoid fee drag while capturing momentum in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 4h calculations
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channel (20-period)
    donchian_period = 20
    upper_channel = np.full_like(high_4h, np.nan)
    lower_channel = np.full_like(low_4h, np.nan)
    
    for i in range(donchian_period - 1, len(high_4h)):
        upper_channel[i] = np.max(high_4h[i - donchian_period + 1:i + 1])
        lower_channel[i] = np.min(low_4h[i - donchian_period + 1:i + 1])
    
    # Previous channel values (for breakout detection)
    prev_upper = np.roll(upper_channel, 1)
    prev_lower = np.roll(lower_channel, 1)
    prev_upper[0] = upper_channel[0]
    prev_lower[0] = lower_channel[0]
    
    # ATR for volatility filtering and stop loss
    atr_period = 14
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_4h[0] - low_4h[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= atr_period:
        atr[atr_period] = np.mean(tr[1:atr_period+1])
        for i in range(atr_period + 1, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # 1d EMA trend filter
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.zeros_like(volume)
    for i in range(vol_ma_period, len(volume)):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    # Align higher timeframe data to 4h
    upper_channel_aligned = align_htf_to_ltf(prices, df_4h, prev_upper)
    lower_channel_aligned = align_htf_to_ltf(prices, df_4h, prev_lower)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, atr_period, ema_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper channel with volume and above 1d EMA
            if close[i] > upper_channel_aligned[i] and vol_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume and below 1d EMA
            elif close[i] < lower_channel_aligned[i] and vol_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below lower channel or below 1d EMA
            if close[i] < lower_channel_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper channel or above 1d EMA
            if close[i] > upper_channel_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_VolumeTrend_v3"
timeframe = "4h"
leverage = 1.0