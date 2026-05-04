#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike
# Camarilla pivot levels (R1/S1) from prior 4h act as intraday support/resistance; breakouts with volume
# indicate institutional participation. 4h EMA50 ensures trend alignment to avoid counter-trend trades.
# Volume confirmation (2.0x 20-period EMA) filters weak breakouts. Designed for 1h timeframe
# to target 15-37 trades/year (60-150 total over 4 years) with discrete sizing (0.20).
# Works in bull markets by buying breakouts in uptrends and in bear markets by selling
# breakdowns in downtrends, avoiding range-bound whipsaws. Session filter (08-20 UTC) reduces noise.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot calculation and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels for each 1h bar using prior 4h bar's OHLC
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(n):
        # Need prior 4h bar data (4h bar must be closed)
        if i < 4:  # 4*1h = 4h, need at least one full 4h bar before current 1h bar
            continue
            
        # Get index of prior 4h bar in 4h dataframe
        # 4h bar index = floor(i / 4) - 1 (since we want prior completed 4h bar)
        idx_4h = (i // 4) - 1
        if idx_4h < 0 or idx_4h >= len(df_4h):
            continue
            
        # Calculate Camarilla levels from prior 4h bar
        h_4h = df_4h['high'].iloc[idx_4h]
        l_4h = df_4h['low'].iloc[idx_4h]
        c_4h = df_4h['close'].iloc[idx_4h]
        
        camarilla_r1[i] = c_4h + (h_4h - l_4h) * 1.1 / 12
        camarilla_s1[i] = c_4h - (h_4h - l_4h) * 1.1 / 12
    
    # Volume confirmation: 2.0x 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + price above 4h EMA50 (uptrend)
            if (close[i] > camarilla_r1[i] and volume_spike and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + volume spike + price below 4h EMA50 (downtrend)
            elif (close[i] < camarilla_s1[i] and volume_spike and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price falls below S1 OR price below 4h EMA50 (trend change)
            if close[i] < camarilla_s1[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price rises above R1 OR price above 4h EMA50 (trend change)
            if close[i] > camarilla_r1[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals