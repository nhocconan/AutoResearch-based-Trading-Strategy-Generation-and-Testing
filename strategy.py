#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d volume spike and ADX trend filter.
# Captures mean-reversion at key pivot levels (S1/S2 for long, R1/R2 for short) with confirmation.
# Works in both bull and bear markets by fading overextended moves at institutional levels.
# Target: 25-35 trades/year by requiring confluence of Camarilla touch, volume surge, and ADX > 20.
# Entry: Long at S1/S2 with volume spike and ADX > 20; Short at R1/R2 with volume spike and ADX > 20.
# Exit: Opposite touch (R1 for S1 long, S2 for R2 short) or volume drops below average.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Camarilla, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-period ADX on daily timeframe
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
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
        result[period-1] = np.nansum(data[1:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # Calculate Camarilla levels from previous day
    # Using (H-L) range from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    # R4 = close + range * 1.1/2, R3 = close + range * 1.1/4, R2 = close + range * 1.1/6, R1 = close + range * 1.1/12
    # S1 = close - range * 1.1/12, S2 = close - range * 1.1/6, S3 = close - range * 1.1/4, S4 = close - range * 1.1/2
    camarilla_r1 = prev_close + range_hl * 1.1 / 12
    camarilla_r2 = prev_close + range_hl * 1.1 / 6
    camarilla_s1 = prev_close - range_hl * 1.1 / 12
    camarilla_s2 = prev_close - range_hl * 1.1 / 6
    
    # Volume confirmation using 1d volume
    vol_d = df_1d['volume'].values
    vol_ma_10_1d = pd.Series(vol_d).rolling(window=10, min_periods=10).mean().values
    
    # Align daily data to 4h (wait for daily close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    vol_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_r2_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_s2_aligned[i]) or np.isnan(vol_ma_10_1d_aligned[i])):
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
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        # Volume confirmation: current volume > 1.5x 10-day average
        volume_confirm = vol_current > 1.5 * vol_ma_10_1d_aligned[i]
        
        if position == 0:
            # Enter long at S1/S2 with volume spike in trending market
            if (trending and volume_confirm and 
                (price_close <= camarilla_s1_aligned[i] or price_close <= camarilla_s2_aligned[i])):
                signals[i] = 0.25
                position = 1
            # Enter short at R1/R2 with volume spike in trending market
            elif (trending and volume_confirm and 
                  (price_close >= camarilla_r1_aligned[i] or price_close >= camarilla_r2_aligned[i])):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches R1 (opposite of S1/S2) or volume drops below average
                if price_close >= camarilla_r1_aligned[i]:
                    exit_signal = True
                elif vol_current < vol_ma_10_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price reaches S2 (opposite of R1/R2) or volume drops below average
                if price_close <= camarilla_s2_aligned[i]:
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

name = "4h_Camarilla_Reversal_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0