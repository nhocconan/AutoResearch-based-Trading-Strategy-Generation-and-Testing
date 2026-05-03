#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout + 1d EMA34 Trend Filter + Volume Confirmation
# Camarilla pivot levels identify key support/resistance from prior 1d session.
# Breakout above R3 or below S3 with volume confirmation signals strong momentum.
# 1d EMA34 filter ensures trades align with higher timeframe trend to avoid whipsaws.
# Volume spike (>2.0x 20-period EMA) confirms breakout authenticity.
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag.
# Works in bull/bear markets by trading with the 1d trend during breakouts.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior 1d session
    # R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each 1d bar
    pivot_range = high_1d - low_1d
    r3_level = close_1d + 1.1 * pivot_range * 1.1 / 4
    s3_level = close_1d - 1.1 * pivot_range * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (available after 1d bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla R3/S3 breakout signals with 1d trend filter
        # Long: price breaks above R3 + price above 1d EMA34 + volume spike
        # Short: price breaks below S3 + price below 1d EMA34 + volume spike
        if position == 0:
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                close[i] > ema_34_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price drops below S3 (mean reversion) OR price below 1d EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 (mean reversion) OR price above 1d EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals