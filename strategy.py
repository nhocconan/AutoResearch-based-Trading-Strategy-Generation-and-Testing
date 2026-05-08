#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h ADX and 1d Williams %R for regime detection and mean reversion entries.
# Long when 12h ADX > 25 (trending) and 1d Williams %R < -80 (oversold) with price > 60-period EMA.
# Short when 12h ADX > 25 (trending) and 1d Williams %R > -20 (overbought) with price < 60-period EMA.
# Exit when ADX < 20 (range) or Williams %R crosses back to neutral (-50).
# Designed for low trade frequency (10-20/year) to avoid fee flood. Works in trending markets with mean reversion entries.

name = "6h_12hADX_1dWilliamsR_TrendMeanRev"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original length
        
        # Directional Movement
        up = high[1:] - high[:-1]
        down = low[:-1] - low[1:]
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smooth TR, +DM, -DM
        atr = np.zeros_like(close)
        plus_dm_smooth = np.zeros_like(close)
        minus_dm_smooth = np.zeros_like(close)
        
        # Wilder's smoothing (first value is simple average)
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
        
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, (plus_dm_smooth / atr) * 100, 0)
        minus_di = np.where(atr != 0, (minus_dm_smooth / atr) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = np.zeros_like(close)
        
        # Wilder's smoothing for ADX
        adx[2*period] = np.nanmean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        
        for i in range(len(close)):
            if i < period - 1:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        wr = np.where((highest_high - lowest_low) != 0, 
                      (highest_high - close) / (highest_high - lowest_low) * -100, 
                      np.nan)
        return wr
    
    williams_r_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate 60-period EMA on 6h for trend filter
    close_series = pd.Series(close)
    ema_60 = close_series.ewm(span=60, adjust=False, min_periods=60).values
    
    # Align 12h ADX and 1d Williams %R to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # Ensure enough data for ADX and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(williams_r_1d_aligned[i]) or
            np.isnan(ema_60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: ADX > 25 (trending), Williams %R < -80 (oversold), price > EMA60
            if (adx_12h_aligned[i] > 25 and 
                williams_r_1d_aligned[i] < -80 and 
                close[i] > ema_60[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 (trending), Williams %R > -20 (overbought), price < EMA60
            elif (adx_12h_aligned[i] > 25 and 
                  williams_r_1d_aligned[i] > -20 and 
                  close[i] < ema_60[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 (range) or Williams %R crosses above -50
            if adx_12h_aligned[i] < 20 or williams_r_1d_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX < 20 (range) or Williams %R crosses below -50
            if adx_12h_aligned[i] < 20 or williams_r_1d_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals