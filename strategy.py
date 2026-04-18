#!/usr/bin/env python3
"""
6h Elder Ray Power + Weekly EMA Trend Filter + Volume Spike
Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
Trend: Weekly EMA34 (bullish when price > EMA, bearish when price < EMA)
Entry: Bull Power > 0 and Bear Power < 0 with volume spike in trend direction
Exit: Opposite Elder Ray signal or trend reversal
Designed for low frequency with clear trend-following edge in both bull/bear markets
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
    
    # Get weekly data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend direction
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Elder Ray components (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray signals
        bullish_ray = bull_power[i] > 0 and bear_power[i] > 0  # Both positive = strong bull
        bearish_ray = bull_power[i] < 0 and bear_power[i] < 0  # Both negative = strong bear
        
        price = close[i]
        above_weekly_ema = price > ema_34_1w_aligned[i]
        below_weekly_ema = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: strong bullish Elder Ray, price above weekly EMA, volume spike
            if (bullish_ray and above_weekly_ema and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: strong bearish Elder Ray, price below weekly EMA, volume spike
            elif (bearish_ray and below_weekly_ema and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: bearish Elder Ray forms or price breaks below weekly EMA
            if bearish_ray or below_weekly_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: bullish Elder Ray forms or price breaks above weekly EMA
            if bullish_ray or above_weekly_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Power_WeeklyEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0