#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla R4/S4 levels from prior 1w act as major support/resistance; breakouts with volume
# indicate strong institutional participation. 1w EMA50 ensures alignment with the weekly trend
# to avoid counter-trend trades. Volume confirmation (1.8x 20-period EMA) filters weak breakouts.
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
# Works in bull markets by buying breakouts in uptrends and in bear markets by selling
# breakdowns in downtrends, avoiding range-bound whipsaws.

name = "1d_Camarilla_R4S4_Breakout_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot calculation and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels for each 1d bar using prior 1w bar's OHLC
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(n):
        # Need prior 1w bar data (1w bar must be closed)
        if i < 7:  # Need at least one full 1w bar before current 1d bar
            continue
            
        # Get index of prior 1w bar in 1w dataframe
        # 1w bar index = floor(i / 7) - 1 (since we want prior completed 1w bar)
        idx_1w = (i // 7) - 1
        if idx_1w < 0 or idx_1w >= len(df_1w):
            continue
            
        # Calculate Camarilla levels from prior 1w bar
        h_1w = df_1w['high'].iloc[idx_1w]
        l_1w = df_1w['low'].iloc[idx_1w]
        c_1w = df_1w['close'].iloc[idx_1w]
        
        camarilla_r4[i] = c_1w + (h_1w - l_1w) * 1.1 / 2  # R4
        camarilla_s4[i] = c_1w - (h_1w - l_1w) * 1.1 / 2  # S4
    
    # Volume confirmation: 1.8x 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R4 + volume spike + price above 1w EMA50 (uptrend)
            if (close[i] > camarilla_r4[i] and volume_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + volume spike + price below 1w EMA50 (downtrend)
            elif (close[i] < camarilla_s4[i] and volume_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S4 OR price below 1w EMA50 (trend change)
            if close[i] < camarilla_s4[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R4 OR price above 1w EMA50 (trend change)
            if close[i] > camarilla_r4[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals