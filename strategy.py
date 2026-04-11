#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v1
# Strategy: 12h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels identify key support/resistance. Breakouts with volume
# confirmation in the direction of the 1d ADX trend yield high-probability trades.
# Works in bull (breakouts up in uptrend) and bear (breakouts down in downtrend) markets.
# Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR and DM
    tr_period = 14
    atr = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    # Initial average
    if len(tr) >= tr_period:
        atr[tr_period] = np.nanmean(tr[1:tr_period+1])
        plus_dm_smooth[tr_period] = np.nanmean(plus_dm[1:tr_period+1])
        minus_dm_smooth[tr_period] = np.nanmean(minus_dm[1:tr_period+1])
        
        # Wilder smoothing
        for i in range(tr_period + 1, len(tr)):
            atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (tr_period - 1) + plus_dm[i]) / tr_period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (tr_period - 1) + minus_dm[i]) / tr_period
    
    # Directional Indicators
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    for i in range(tr_period, len(tr)):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX: smoothed DX
    adx = np.full_like(tr, np.nan)
    adx_period = 14
    if len(dx) >= adx_period + tr_period:
        # First ADX value is average of first adx_period DX values
        start_idx = tr_period
        end_idx = start_idx + adx_period
        if end_idx <= len(dx):
            adx[end_idx-1] = np.nanmean(dx[start_idx:end_idx])
            
            # Wilder smoothing for subsequent ADX values
            for i in range(end_idx, len(dx)):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d average volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Camarilla pivot levels from previous 1d candle
    # Typical price = (H + L + C) / 3
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    # Camarilla levels
    # Resistance levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), R2 = C + ((H-L) * 1.1/6), R1 = C + ((H-L) * 1.1/12)
    # Support levels: S1 = C - ((H-L) * 1.1/12), S2 = C - ((H-L) * 1.1/6), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_r2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_s2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Warmup period
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x average 1d volume
        vol_confirm = volume[i] > (1.5 * vol_avg_1d_aligned[i])
        
        # Long entry: price breaks above R3 with volume and strong trend
        if (close[i] > camarilla_r3_aligned[i] and vol_confirm and strong_trend and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below S3 with volume and strong trend
        elif (close[i] < camarilla_s3_aligned[i] and vol_confirm and strong_trend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to mean (S1/R1) or trend weakens
        elif position == 1 and (close[i] < camarilla_s1_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > camarilla_r1_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals