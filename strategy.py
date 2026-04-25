#!/usr/bin/env python3
"""
6h Weekly Donchian Breakout with Daily Camarilla Filter and Volume Spike
Hypothesis: Weekly Donchian channels identify major structural breaks. Combined with daily Camarilla R3/S3 levels for directional bias and volume confirmation, this captures strong momentum moves while avoiding false breakouts. Works in bull/bear via trend filter and uses discrete sizing (0.25) to limit fee drag (~75-150 trades over 4 years).
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
    
    # Get weekly data for Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period high/low)
    high_1w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_1w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R3, S3, R4, S4)
    # Camarilla formula: 
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # Where C = (H+L+Close)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    
    camarilla_r4 = typical_price + (hl_range * 1.1 / 2)
    camarilla_r3 = typical_price + (hl_range * 1.1 / 4)
    camarilla_s3 = typical_price - (hl_range * 1.1 / 4)
    camarilla_s4 = typical_price - (hl_range * 1.1 / 2)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Calculate ATR for volume normalization and stop reference
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly Donchian (20) and daily data
    start_idx = max(20, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Get aligned levels
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        camarilla_r3 = camarilla_r3_aligned[i]
        camarilla_s3 = camarilla_s3_aligned[i]
        camarilla_r4 = camarilla_r4_aligned[i]
        camarilla_s4 = camarilla_s4_aligned[i]
        
        # Breakout conditions: price breaks weekly Donchian channel
        bullish_breakout = curr_close > donch_high
        bearish_breakout = curr_close < donch_low
        
        # Camarilla filter: 
        # Long: price above R3 (bullish bias) AND not at extreme R4 (avoid exhaustion)
        # Short: price below S3 (bearish bias) AND not at extreme S4 (avoid exhaustion)
        long_bias = curr_close > camarilla_r3 and curr_close < camarilla_r4
        short_bias = curr_close < camarilla_s3 and curr_close > camarilla_s4
        
        # Exit conditions: reverse breakout or volatility contraction
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout or if price drops below Camarilla S3
                if bearish_breakout or curr_close < camarilla_s3:
                    exit_signal = True
            elif position == -1:
                # Exit on bullish breakout or if price rises above Camarilla R3
                if bullish_breakout or curr_close > camarilla_r3:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Weekly Donchian breakout + Camarilla bias + volume spike
        if position == 0:
            # Long: break above weekly Donchian high AND long bias AND volume spike
            long_condition = bullish_breakout and long_bias and volume_spike
            # Short: break below weekly Donchian low AND short bias AND volume spike
            short_condition = bearish_breakout and short_bias and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian_Breakout_DailyCamarilla_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0