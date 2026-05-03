#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index + 1d EMA50 trend + volume confirmation
# Elder Ray measures bull/bear power vs EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Bullish when Bull Power > 0 AND Bear Power < previous Bear Power (bulls gaining control)
# Bearish when Bear Power < 0 AND Bull Power < previous Bull Power (bears gaining control)
# 1d EMA50 ensures alignment with higher timeframe trend
# Volume spike (>2.0x 20-period EMA) confirms institutional participation
# Works in bull/bear: Elder Ray identifies power shifts, volume filters weak moves,
# HTF EMA prevents counter-trend trading
# Target: 60-100 total trades over 4 years (15-25/year) to stay within fee drag limits

name = "6h_ElderRay_1dEMA50_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray Index: Bull Power and Bear Power vs EMA13
    # EMA13 on typical price
    typical_price = (high + low + close) / 3.0
    ema13 = pd.Series(typical_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid EMA13 values
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to reduce trades)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Elder Ray signals with 1d trend filter
        # Long: Bull Power > 0 AND Bear Power < previous Bear Power (bulls gaining) 
        #       + price above 1d EMA50 + volume spike
        # Short: Bear Power < 0 AND Bull Power < previous Bull Power (bears gaining)
        #        + price below 1d EMA50 + volume spike
        if position == 0:
            if (bull_power[i] > 0 and bear_power[i] < bear_power[i-1] and 
                close[i] > ema_50_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (bear_power[i] < 0 and bull_power[i] < bull_power[i-1] and 
                  close[i] < ema_50_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power turns positive OR price below 1d EMA50
            if bear_power[i] > 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power turns negative OR price above 1d EMA50
            if bull_power[i] < 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals