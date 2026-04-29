#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d ADX trend filter and volume spike confirmation
# Camarilla R3/S3 levels act as strong intraday support/resistance. Breakouts above R3 or below S3
# with volume spike and ADX > 25 indicate strong momentum continuation. Works in both bull and bear
# markets by capturing breakout moves. Target: 12-30 trades/year (50-120 total).

name = "6h_Camarilla_R3S3_Breakout_1dADX25_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    # ADX requires +DI, -DI, and TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    tr_period = 14
    tr_smoothed = wilders_smoothing(tr, tr_period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, tr_period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, tr_period)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, tr_period)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: based on previous day's range
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    daily_range = prev_day_high - prev_day_low
    camarilla_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    r4 = camarilla_pivot + (daily_range * 1.1 / 2)
    r3 = camarilla_pivot + (daily_range * 1.1 / 4)
    r2 = camarilla_pivot + (daily_range * 1.1 / 6)
    r1 = camarilla_pivot + (daily_range * 1.1 / 12)
    s1 = camarilla_pivot - (daily_range * 1.1 / 12)
    s2 = camarilla_pivot - (daily_range * 1.1 / 6)
    s3 = camarilla_pivot - (daily_range * 1.1 / 4)
    s4 = camarilla_pivot - (daily_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 2.0x 20-period average (strong breakout)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 1d ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_adx = adx_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        
        # Require strong trend (ADX > 25) and volume confirmation for breakout trades
        strong_trend_and_volume = (curr_adx > 25) and curr_volume_confirm
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout above R3 with strong trend and volume
            if curr_close > curr_r3 and strong_trend_and_volume:
                signals[i] = 0.25
                position = 1
            # Short breakdown below S3 with strong trend and volume
            elif curr_close < curr_s3 and strong_trend_and_volume:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to pivot (mean reversion) OR breaks R4 (take profit)
            if curr_close < curr_pivot or curr_close > curr_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to pivot (mean reversion) OR breaks S4 (take profit)
            if curr_close > curr_pivot or curr_close < curr_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals