#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla levels provide precise intraday support/resistance for breakouts
# 4h EMA50 ensures alignment with intermediate trend to avoid counter-trend trades
# Volume spike (>2.0x 20-period EMA) filters low-probability breakouts
# Session filter (08-20 UTC) reduces noise trades
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag
# Works in bull/bear: trend filter ensures we trade with higher timeframe momentum

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session"
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
    
    # Calculate 4h data for volume EMA
    volume_4h = df_4h['volume'].values
    vol_ema_20_4h = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20_4h)
    
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
    
    camarilla_r3 = close_prev + 1.1 * (high_prev - low_prev)
    camarilla_s3 = close_prev - 1.1 * (high_prev - low_prev)
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i]) or
            np.isnan(vol_ema_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # 4h volume confirmation: current 4h volume > 2.0 x 20-period 4h EMA
        volume_spike_4h = False
        if i >= 4:  # Need at least 4 1h bars to align with 4h bar
            volume_spike_4h = volume[i] > (2.0 * vol_ema_20_4h_aligned[i])
        
        # Combined volume confirmation (either 1h or 4h spike)
        volume_confirmed = volume_spike or volume_spike_4h
        
        # Camarilla breakout signals with 4h trend filter
        # Long: Break above R3 + price above 4h EMA50 + volume confirmation + session
        # Short: Break below S3 + price below 4h EMA50 + volume confirmation + session
        if position == 0:
            if in_session[i] and volume_confirmed:
                if close[i] > camarilla_r3[i] and close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                elif close[i] < camarilla_s3[i] and close[i] < ema_50_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 (reversion to mean) OR below 4h EMA50
            if close[i] < camarilla_s3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above R3 (reversion to mean) OR above 4h EMA50
            if close[i] > camarilla_r3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals