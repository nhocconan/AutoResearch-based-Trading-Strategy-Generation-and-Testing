#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Camarilla R1/S1 breakout + volume spike + 4h EMA34 trend filter.
Long when price breaks above R1 with volume > 2x 20-period average and 4h EMA34 rising.
Short when price breaks below S1 with volume > 2x 20-period average and 4h EMA34 falling.
Uses Camarilla pivot levels from 1d for intraday support/resistance, volume confirmation for conviction,
and 4h EMA34 for trend alignment. Designed to capture mean-reversion breaks in ranging markets
and trend continuation in trending markets, working in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    hl_range = high_1d - low_1d
    camarilla_r1 = close_1d + hl_range * 1.1 / 12
    camarilla_s1 = close_1d - hl_range * 1.1 / 12
    
    # Calculate 4h EMA34 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume average on 12h timeframe
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (12h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)  # align 1d volume MA to 12h
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # need enough for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        # EMA34 trend: rising if current > previous, falling if current < previous
        if i > 0:
            ema_rising = ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1]
            ema_falling = ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1]
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and rising EMA34
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_confirmed and 
                ema_rising):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and falling EMA34
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_confirmed and 
                  ema_falling):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S1 (opposite side of pivot)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R1 (opposite side of pivot)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarilla_R1S1_VolumeSpike_4hEMA34_Trend"
timeframe = "12h"
leverage = 1.0