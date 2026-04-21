#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot point reversal with 1d volume surge and ADX trend filter.
# Works in bull/bear: mean reversion at key levels with volume confirmation and trend alignment.
# Target: 20-30 trades/year by requiring tight confluence of Camarilla touch, volume spike, and trend filter.
# Entry: Long at S1 when price touches S1 with volume surge and ADX > 20 (bullish bias); Short at R1 when price touches R1 with volume surge and ADX > 20 (bearish bias).
# Exit: Opposite touch of Camarilla level (S3 for long, R3 for short) or volume drops below average.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Camarilla calculation, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-period ADX on daily timeframe for trend strength
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # +DM and -DM
    up_move = high_d[1:] - high_d[:-1]
    down_move = low_d[:-1] - low_d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Wilder's smoothing function
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average (skip first NaN in TR)
        result[period-1] = np.nansum(data[1:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L), etc.
    # We'll use: R1 = C + 1.1*(H-L)/2, S1 = C - 1.1*(H-L)/2
    #          R3 = C + 1.1*(H-L), S3 = C - 1.1*(H-L)
    #          R4 = C + 1.5*(H-L), S4 = C - 1.5*(H-L)
    # But we need previous day's values, so we shift by 1
    prev_close = np.concatenate([[np.nan], close_d[:-1]])
    prev_high = np.concatenate([[np.nan], high_d[:-1]])
    prev_low = np.concatenate([[np.nan], low_d[:-1]])
    
    hl_range = prev_high - prev_low
    camarilla_S1 = prev_close - 1.1 * hl_range / 2
    camarilla_R1 = prev_close + 1.1 * hl_range / 2
    camarilla_S3 = prev_close - 1.1 * hl_range
    camarilla_R3 = prev_close + 1.1 * hl_range
    
    # Volume confirmation using 1d volume
    vol_d = df_1d['volume'].values
    vol_ma_10_1d = pd.Series(vol_d).rolling(window=10, min_periods=10).mean().values
    
    # Align daily data to 4h (wait for daily close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    vol_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(vol_ma_10_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        vol_current = align_htf_to_ltf(prices, df_1d, vol_d)[i]  # 1d volume aligned to 4h
        
        # Camarilla levels
        s1 = camarilla_S1_aligned[i]
        r1 = camarilla_R1_aligned[i]
        s3 = camarilla_S3_aligned[i]
        r3 = camarilla_R3_aligned[i]
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        # Volume confirmation: current volume > 1.5x 10-day average
        volume_confirm = vol_current > 1.5 * vol_ma_10_1d_aligned[i]
        
        if position == 0:
            # Enter long: price touches S1 with volume surge in trending market (bullish bias)
            if (trending and 
                price_close <= s1 and  # Touch or penetrate S1
                price_close >= s3 and  # But not below S3 (avoid strong breakdown)
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches R1 with volume surge in trending market (bearish bias)
            elif (trending and 
                  price_close >= r1 and  # Touch or penetrate R1
                  price_close <= r3 and  # But not above R3 (avoid strong breakout)
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches S3 (strong support break) OR volume drops below average
                if price_close <= s3:
                    exit_signal = True
                elif vol_current < vol_ma_10_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price touches R3 (strong resistance break) OR volume drops below average
                if price_close >= r3:
                    exit_signal = True
                elif vol_current < vol_ma_10_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_VolumeSurge_ADX"
timeframe = "4h"
leverage = 1.0