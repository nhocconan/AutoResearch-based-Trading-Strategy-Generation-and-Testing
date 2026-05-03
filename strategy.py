#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivots identify key support/resistance levels. Breakouts at R3/S3 with 1d EMA34 trend
# alignment and volume spike capture institutional participation. Designed for 12-30 trades/year on 6h
# to minimize fee drag while maintaining edge in both bull and bear markets via trend-following breakouts.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # We need previous day's data to calculate today's Camarilla levels
    # For each 6h bar, we use the most recent completed 1d bar's OHLC
    for i in range(n):
        if i == 0:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            continue
            
        # Find the index of the most recent completed 1d bar
        # Since we're on 6h timeframe, we need to map to 1d bars
        # We'll use the previous day's close for simplicity (conservative)
        if i >= 4:  # At least 4 six-hour bars = 1 day
            # Use OHLC from 4 bars ago (yesterday's close, high, low, open)
            idx_lookback = i - 4
            if idx_lookback >= 0 and idx_lookback < n:
                # We need the actual 1d OHLC, so we'll use the HTF data
                # Find corresponding 1d bar index
                # Simpler approach: use rolling window on 1d data
                pass
    
    # Instead, calculate Camarilla levels directly from 1d data and align
    # Camarilla formula: based on previous day's (high, low, close)
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate Camarilla levels for each 1d bar (based on previous day)
        camarilla_r3_1d = np.full(len(df_1d), np.nan)
        camarilla_s3_1d = np.full(len(df_1d), np.nan)
        
        for j in range(1, len(df_1d)):
            # Previous day's OHLC
            phigh = high_1d[j-1]
            plow = low_1d[j-1]
            pclose = close_1d[j-1]
            
            # Camarilla levels
            camarilla_r3_1d[j] = pclose + (phigh - plow) * 1.1 / 4
            camarilla_s3_1d[j] = pclose - (phigh - plow) * 1.1 / 4
        
        # Align to 6h timeframe
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d, additional_delay_bars=1)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d, additional_delay_bars=1)
    else:
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(4, n):  # Start after sufficient warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r3_aligned[i]
        breakout_short = close[i] < camarilla_s3_aligned[i]
        
        # Trend filter: 1d EMA34 direction
        # For long: price above EMA34 (uptrend)
        # For short: price below EMA34 (downtrend)
        trend_long = close[i] > ema_34_1d_aligned[i]
        trend_short = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above R3 with uptrend and volume spike
            if breakout_long and trend_long and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 with downtrend and volume spike
            elif breakout_short and trend_short and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakout fails or trend changes
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout fails or trend changes
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals