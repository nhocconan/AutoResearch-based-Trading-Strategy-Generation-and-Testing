#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 Trend Filter + Volume Spike
# Long when Alligator jaws (13) < teeth (8) < lips (5) AND price > 1d EMA34 AND volume spike
# Short when Alligator jaws (13) > teeth (8) > lips (5) AND price < 1d EMA34 AND volume spike
# Williams Alligator identifies trend alignment: jaws (13-period SMMA), teeth (8-period SMMA), lips (5-period SMMA)
# When lips > teeth > jaws = uptrend, lips < teeth < jaws = downtrend
# 1d EMA34 provides higher-timeframe trend filter to avoid counter-trend trades
# Volume spike requires 2.0x 20-bar MA for confirmation (reduces false signals)
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag while capturing strong trends
# Works in bull (trend alignment + breakouts) and bear (strong downtrends with volume)
# Timeframe: 12h (as required)

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    median_price_12h = (high_12h + low_12h + close_12h) / 3.0  # Typical price
    
    # Calculate SMMA (Smoothed Moving Average) - Williams Alligator uses SMMA
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Alligator lines: Jaws (13), Teeth (8), Lips (5)
    jaws = smma(median_price_12h, 13)  # Blue line (13-period)
    teeth = smma(median_price_12h, 8)   # Red line (8-period)
    lips = smma(median_price_12h, 5)    # Green line (5-period)
    
    # Align Alligator lines to 12h timeframe (they're already on 12h, but align for safety)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume confirmation on 12h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data for SMMA)
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaws (uptrend alignment) AND price > 1d EMA34 AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaws_aligned[i] and
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaws (downtrend alignment) AND price < 1d EMA34 AND volume spike
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaws_aligned[i] and
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend breaks down (Lips <= Teeth OR Teeth <= Jaws) OR price < 1d EMA34
            if (lips_aligned[i] <= teeth_aligned[i] or 
                teeth_aligned[i] <= jaws_aligned[i] or
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend breaks up (Lips >= Teeth OR Teeth >= Jaws) OR price > 1d EMA34
            if (lips_aligned[i] >= teeth_aligned[i] or 
                teeth_aligned[i] >= jaws_aligned[i] or
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals