#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA(34) trend filter and volume confirmation
# Camarilla pivot levels from 1d provide high-probability support/resistance levels
# Breakout above R1 or below S1 with volume confirmation captures institutional moves
# 1w EMA(34) ensures alignment with weekly trend to avoid counter-trend trades in bear markets
# Volume spike (>2.0x 20-period EMA) filters low-probability breakouts
# Works in bull/bear markets by following 1w trend direction for entries
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag and maximize edge

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_VolumeSpike"
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
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivot levels from 1d data (standard formula)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = (high_1d - low_1d)
    r1 = close_1d + (1.1 * camarilla_range / 12)
    s1 = close_1d - (1.1 * camarilla_range / 12)
    
    # Align Camarilla levels to 1d timeframe (already aligned as 1d->1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout signals with 1w trend filter
        # Long: Break above R1 + price above 1w EMA34 + volume spike
        # Short: Break below S1 + price below 1w EMA34 + volume spike
        if position == 0:
            if close[i] > r1_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S1 (reversion to mean) OR below 1w EMA34
            if close[i] < s1_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R1 (reversion to mean) OR above 1w EMA34
            if close[i] > r1_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals