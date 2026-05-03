#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla R1/S1 provides tighter intraday levels for higher probability breakouts
# 4h EMA50 ensures alignment with medium-term trend to avoid counter-trend trades
# Volume spike (>1.8x 20-period EMA) filters low-probability breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Target: 80-120 total trades over 4 years (20-30/year) to balance edge and fee drag

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1h bar
    typical_price = (high + low + close) / 3.0
    typical_price_prev = np.roll(typical_price, 1)
    typical_price_prev[0] = np.nan
    
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_r1 = close_prev + 1.1 * (high_prev - low_prev) * (1.0/6.0)  # R1 = close + 1.1*(high-low)/6
    camarilla_s1 = close_prev - 1.1 * (high_prev - low_prev) * (1.0/6.0)  # S1 = close - 1.1*(high-low)/6
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade during 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0 and in_session:
            # Long: Break above R1 + price above 4h EMA50 + volume spike
            if close[i] > camarilla_r1[i] and close[i] > ema_50_4h_aligned[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: Break below S1 + price below 4h EMA50 + volume spike
            elif close[i] < camarilla_s1[i] and close[i] < ema_50_4h_aligned[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S1 OR below 4h EMA50
            if close[i] < camarilla_s1[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above R1 OR above 4h EMA50
            if close[i] > camarilla_r1[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Outside session or no signal: maintain flat or current position
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals