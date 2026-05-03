#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla levels provide precise daily support/resistance for breakouts
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades
# Volume spike (>2.0x 20-period EMA) filters low-probability breakouts
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag and improve generalization

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Shift to use previous bar's typical price (no look-ahead)
    typical_price_prev = np.roll(typical_price, 1)
    typical_price_prev[0] = np.nan  # First bar has no previous
    
    # Camarilla R3, S3 levels based on previous bar
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_r3 = close_prev + 1.1 * (high_prev - low_prev)
    camarilla_s3 = close_prev - 1.1 * (high_prev - low_prev)
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout signals with 1w trend filter
        # Long: Break above R3 + price above 1w EMA50 + volume spike
        # Short: Break below S3 + price below 1w EMA50 + volume spike
        if position == 0:
            if close[i] > camarilla_r3[i] and close[i] > ema_50_1w_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif close[i] < camarilla_s3[i] and close[i] < ema_50_1w_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 (reversion to mean) OR below 1w EMA50
            if close[i] < camarilla_s3[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R3 (reversion to mean) OR above 1w EMA50
            if close[i] > camarilla_r3[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals