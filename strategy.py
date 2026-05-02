#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter
# Uses Camarilla pivot levels from prior 1d bar for structure-based entries
# R3/S3 levels act as strong support/resistance - breaks indicate momentum continuation
# 1d volume spike (2.0x 20-period avg) confirms institutional participation
# ADX(14) > 25 on 1d ensures trending market to avoid choppy whipsaws
# Discrete sizing 0.25 targets 75-150 trades over 4 years (19-38/year)
# Works in bull/bear by only taking breaks with volume and trend confirmation

name = "4h_Camarilla_R3S3_Breakout_1dVolume_ADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    
    # Camarilla R3, S3 levels (primary breakout levels)
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d volume confirmation (2.0x 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike_1d = df_1d['volume'].values > (vol_ma_1d * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1d ADX(14) for trend regime filter
    # ADX calculation: +DI, -DI, DX, then ADX
    high_1d_series = pd.Series(df_1d['high'])
    low_1d_series = pd.Series(df_1d['low'])
    close_1d_series = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1d_series - high_1d_series.shift(1)
    down_move = low_1d_series.shift(1) - low_1d_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_1d_values = adx_1d.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for ADX and volume calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Camarilla R3 breakout long: close > R3 with volume spike and ADX > 25
            breakout_long = close[i] > camarilla_r3_aligned[i]
            # Camarilla S3 breakdown short: close < S3 with volume spike and ADX > 25
            breakout_short = close[i] < camarilla_s3_aligned[i]
            
            # Regime filters
            vol_confirm = volume_spike_1d_aligned[i]
            trend_strong = adx_1d_aligned[i] > 25
            
            if breakout_long and vol_confirm and trend_strong:
                signals[i] = 0.25
                position = 1
            elif breakout_short and vol_confirm and trend_strong:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Camarilla S3 breakdown or loss of momentum (ADX < 20)
            if close[i] < camarilla_s3_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Camarilla R3 breakout or loss of momentum (ADX < 20)
            if close[i] > camarilla_r3_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals