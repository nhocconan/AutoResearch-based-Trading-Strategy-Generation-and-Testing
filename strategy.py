#!/usr/bin/env python3
# Hypothesis: 4h Camarilla pivot level breakouts with volume confirmation and ADX trend filter.
# Uses Camarilla levels from daily timeframe (resistance/support levels) for precise entries.
# ADX > 25 filters for trending markets, reducing whipsaws in sideways action.
# Volume confirmation ensures breakouts have institutional participation.
# Designed to work in both bull and bear markets by trading breakouts in direction of trend.
# Target: 20-40 trades/year per symbol with controlled risk.

name = "4H_Camarilla_Pivot_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla levels and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    # Camarilla: Range = H-L, then levels at specific fractions
    daily_range = high_1d - low_1d
    camarilla_r4 = close_1d + daily_range * 1.1 / 2  # Resistance 4
    camarilla_r3 = close_1d + daily_range * 1.1 / 4  # Resistance 3
    camarilla_r2 = close_1d + daily_range * 1.1 / 6  # Resistance 2
    camarilla_r1 = close_1d + daily_range * 1.1 / 12 # Resistance 1
    camarilla_s1 = close_1d - daily_range * 1.1 / 12 # Support 1
    camarilla_s2 = close_1d - daily_range * 1.1 / 6  # Support 2
    camarilla_s3 = close_1d - daily_range * 1.1 / 4  # Support 3
    camarilla_s4 = close_1d - daily_range * 1.1 / 2  # Support 4
    
    # Calculate ADX for trend strength (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(tr)
        minus_di = np.zeros_like(tr)
        
        # Wilder's smoothing
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_sum = np.nansum(plus_dm[1:period+1])
        minus_dm_sum = np.nansum(minus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum * (period-1) / period + plus_dm[i]
            minus_dm_sum = minus_dm_sum * (period-1) / period + minus_dm[i]
        
        # Avoid division by zero
        atr[atr == 0] = 1e-10
        
        plus_di = 100 * plus_dm_sum / atr
        minus_di = 100 * minus_dm_sum / atr
        
        dx = np.zeros_like(tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.zeros_like(tr)
        
        # Smooth DX to get ADX
        adx[2*period] = np.nanmean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(tr)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align all daily indicators to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average volume
        if i >= 20:
            avg_volume = np.mean(volume[max(0, i-20):i])
            volume_confirm = volume[i] > avg_volume * 1.5
        else:
            volume_confirm = False
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Enter long: price breaks above Camarilla R3 with volume and trend
            if (close[i] > camarilla_r3_aligned[i] and volume_confirm and trend_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S3 with volume and trend
            elif (close[i] < camarilla_s3_aligned[i] and volume_confirm and trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Camarilla R1 or trend weakens
            if (close[i] < camarilla_r1_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Camarilla S1 or trend weakens
            if (close[i] > camarilla_s1_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals