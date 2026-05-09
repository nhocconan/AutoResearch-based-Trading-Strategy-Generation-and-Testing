#!/usr/bin/env python3
# Hypothesis: 1d Weekly Pivot (R3/S3) breakout with volume confirmation and ADX filter for trend strength
# Works in bull markets by catching breakouts above weekly R3 or below weekly S3
# Works in bear markets by fading false breaks when price reverses back to weekly pivot (mean reversion)
# Uses weekly high/low/close to calculate Camarilla levels R3/S3
# Long when: close > weekly R3, volume spike (>1.5x 20-day average), ADX > 25 (trending market)
# Short when: close < weekly S3, volume spike, ADX > 25
# Exit when: price crosses back to weekly pivot (mean reversion) OR ADX < 20 (range market)
# Position size: 0.25 to limit drawdown. Target: 15-25 trades/year.

name = "1d_WeeklyCamarilla_R3S3_Volume_ADX"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week's high, low, close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels R3 and S3
    camarilla_r3 = close_1w + ((high_1w - low_1w) * 1.2500)
    camarilla_s3 = close_1w - ((high_1w - low_1w) * 1.2500)
    camarilla_pivot = (camarilla_r3 + camarilla_s3) / 2  # Midpoint for exit
    
    # Align weekly Camarilla levels to daily timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Volume spike: current volume > 1.5x 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    # ADX calculation for trend strength (using daily data)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = previous * (1 - 1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > weekly R3 + volume spike + ADX > 25 (trending)
            if (close[i] > camarilla_r3_aligned[i] and 
                vol_spike[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Enter short: price < weekly S3 + volume spike + ADX > 25 (trending)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  vol_spike[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly pivot OR ADX < 20 (range market)
            if (close[i] < camarilla_pivot_aligned[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly pivot OR ADX < 20 (range market)
            if (close[i] > camarilla_pivot_aligned[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals