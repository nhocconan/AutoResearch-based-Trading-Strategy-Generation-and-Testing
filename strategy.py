#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level with 1d uptrend (ADX > 25 and +DI > -DI) and volume > 2.0x 20-bar avg.
# Short when price breaks below Camarilla S3 level with 1d downtrend (ADX > 25 and -DI > +DI) and volume > 2.0x 20-bar avg.
# Exit on opposite Camarilla level touch (mean reversion within the pivot structure).
# Uses proven Camarilla pivot structure with strict volume confirmation (2.0x) and 1d ADX trend filter to limit trades.
# 1d ADX > 25 ensures we only trade in strong trending markets, reducing false signals in choppy markets and bear rallies.
# Timeframe: 6h, HTF: 1d as per experiment guidelines.

name = "6h_Camarilla_R3S3_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize smoothed values
    atr = np.full_like(tr, np.nan)
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    adx = np.full_like(tr, np.nan)
    
    # First ATR is simple average of first 'period' TR values
    if len(tr) >= period + 1:
        atr[period] = np.nanmean(tr[1:period+1])
        # First smoothed DM values
        plus_dm_smooth = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth = np.nansum(minus_dm[1:period+1])
        
        # First DI values
        if atr[period] != 0:
            plus_di[period] = (plus_dm_smooth / atr[period]) * 100
            minus_di[period] = (minus_dm_smooth / atr[period]) * 100
            dx[period] = (np.abs(plus_di[period] - minus_di[period]) / (plus_di[period] + minus_di[period])) * 100 if (plus_di[period] + minus_di[period]) != 0 else 0
        
        # Wilder's smoothing for subsequent values
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            plus_dm_smooth = (plus_dm_smooth * (period - 1) + plus_dm[i]) / period
            minus_dm_smooth = (minus_dm_smooth * (period - 1) + minus_dm[i]) / period
            if atr[i] != 0:
                plus_di[i] = (plus_dm_smooth / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth / atr[i]) * 100
                dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # ADX is smoothed DX
        adx_period = period
        if len(dx) >= 2 * adx_period:
            adx[2*adx_period-1] = np.nanmean(dx[adx_period:2*adx_period])
            for i in range(2*adx_period, len(dx)):
                adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align 1d ADX, +DI, -DI to 6h timeframe (completed 1d bar only)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Previous 1d OHLC for completed 1d bar (no look-ahead)
    df_1d_prev = get_htf_data(prices, '1d')
    if len(df_1d_prev) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d_prev['high'].shift(1).values
    prev_low_1d = df_1d_prev['low'].shift(1).values
    prev_close_1d = df_1d_prev['close'].shift(1).values
    
    # Align 1d data to 6h timeframe (completed 1d bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_low_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_close_1d)
    
    # Camarilla pivot levels from previous completed 1d bar (no look-ahead)
    # R3 = close + 1.1*(high - low)/2, S3 = close - 1.1*(high - low)/2
    camarilla_r3 = prev_close_aligned + 1.1 * (prev_high_aligned - prev_low_aligned) / 2
    camarilla_s3 = prev_close_aligned - 1.1 * (prev_high_aligned - prev_low_aligned) / 2
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for ADX and Camarilla
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(plus_di_1d_aligned[i]) or np.isnan(minus_di_1d_aligned[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_camarilla_r3 = camarilla_r3[i]
        curr_camarilla_s3 = camarilla_s3[i]
        curr_adx_1d = adx_1d_aligned[i]
        curr_plus_di_1d = plus_di_1d_aligned[i]
        curr_minus_di_1d = minus_di_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine 1d trend: ADX > 25 and directional bias
        is_uptrend = curr_adx_1d > 25 and curr_plus_di_1d > curr_minus_di_1d
        is_downtrend = curr_adx_1d > 25 and curr_minus_di_1d > curr_plus_di_1d
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, uptrend, volume spike
            if (curr_close > curr_camarilla_r3 and 
                is_uptrend and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, downtrend, volume spike
            elif (curr_close < curr_camarilla_s3 and 
                  is_downtrend and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Camarilla S3 (mean reversion)
            if curr_close <= curr_camarilla_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Camarilla R3 (mean reversion)
            if curr_close >= curr_camarilla_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals