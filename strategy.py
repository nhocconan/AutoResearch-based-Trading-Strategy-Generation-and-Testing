# 101599: 6h Camarilla R3/S3 Breakout with 12h Trend and Volume Spike
# Uses Camarilla pivot levels from daily timeframe for structured support/resistance
# Breakouts at R3/S3 with trend confirmation and volume spike for institutional participation
# Designed to work in both bull (breakouts continue) and bear (mean reversion at extremes) markets
# Target: 20-50 trades/year (~80-200 over 4 years) to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # Using previous day's OHLC to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First day will have rolled values, but we'll mask invalid later
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_r4 = close_1d_prev + ((high_1d_prev - low_1d_prev) * 1.1 / 2)
    camarilla_r3 = close_1d_prev + ((high_1d_prev - low_1d_prev) * 1.1 / 4)
    camarilla_s3 = close_1d_prev - ((high_1d_prev - low_1d_prev) * 1.1 / 4)
    camarilla_s4 = close_1d_prev - ((high_1d_prev - low_1d_prev) * 1.1 / 2)
    
    # Align to 6h timeframe
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA(20) for trend
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_6h[i]) or 
            np.isnan(camarilla_s3_6h[i]) or
            np.isnan(ema_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA20
        price_above_ema = close[i] > ema_20_12h_aligned[i]
        price_below_ema = close[i] < ema_20_12h_aligned[i]
        
        # Breakout at R3/S3 with volume spike
        breakout_r3 = close[i] > camarilla_r3_6h[i]
        breakdown_s3 = close[i] < camarilla_s3_6h[i]
        
        # Long conditions: bullish trend + volume spike + R3 breakout
        long_condition = (price_above_ema and 
                         volume_spike[i] and 
                         breakout_r3)
        
        # Short conditions: bearish trend + volume spike + S3 breakdown
        short_condition = (price_below_ema and 
                          volume_spike[i] and 
                          breakdown_s3)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal OR price reaches R4/S4 (extreme levels)
        elif position == 1 and (not price_above_ema or close[i] >= camarilla_r4_6h[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or close[i] <= camarilla_s4_6h[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0