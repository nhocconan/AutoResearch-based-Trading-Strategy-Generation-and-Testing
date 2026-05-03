#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot points identify key support/resistance levels; breakouts above R3 or below S3 indicate strong momentum
# 4h EMA50 ensures we trade with higher timeframe trend to avoid whipsaws and false breakouts
# Volume spike (>1.8x 24-period EMA) confirms breakout authenticity and filters low-conviction moves
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag
# Works in bull/bear: trend filter adapts to market direction, volume confirmation ensures quality

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike"
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
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime operations in loop)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivot points (R3, S3 levels)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    pivot = (highest_high + lowest_low + close) / 3.0
    range_hl = highest_high - lowest_low
    
    # Camarilla levels: R3 = close + range_hl * 1.1/4, S3 = close - range_hl * 1.1/4
    camarilla_r3 = close + (range_hl * 1.1 / 4.0)
    camarilla_s3 = close - (range_hl * 1.1 / 4.0)
    
    # Volume confirmation: 24-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_24 = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ema_24[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 24-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_24[i])
        
        # Camarilla R3/S3 breakout signals with 4h trend filter
        # Long: price breaks above R3 + price above 4h EMA50 + volume spike
        # Short: price breaks below S3 + price below 4h EMA50 + volume spike
        if position == 0:
            if (close[i] > camarilla_r3[i] and close[i] > ema_50_4h_aligned[i] and volume_spike):
                signals[i] = 0.20
                position = 1
            elif (close[i] < camarilla_s3[i] and close[i] < ema_50_4h_aligned[i] and volume_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price falls below pivot (mean reversion) OR below 4h EMA50 (trend change)
            if close[i] < pivot[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price rises above pivot (mean reversion) OR above 4h EMA50 (trend change)
            if close[i] > pivot[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals