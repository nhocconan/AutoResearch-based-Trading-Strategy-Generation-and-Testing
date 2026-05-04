#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d trend filter and volume confirmation
# Uses Alligator lines (SMAs) to identify trend direction and entry when price crosses Lips with confirmation.
# 1d EMA50 trend filter avoids counter-trend trades. Volume spike confirms momentum.
# Discrete position sizing (0.25) minimizes fee churn. Target: 12-25 trades/year per symbol.
# Works in bull markets (trend following) and bear markets (avoids shorts in strong downtrends via 1d filter).

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h data for Williams Alligator (SMAs: Jaw=13, Teeth=8, Lips=5)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Alligator lines (all based on median price)
    median_price = (df_12h['high'] + df_12h['low']) / 2
    close_12h = median_price.values
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 12h data for volume EMA(20) for volume confirmation
    vol_12h = df_12h['volume'].values
    vol_ema_20 = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        # 1d trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_1d_aligned[i]
        bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price crosses above Lips + volume confirmation + bullish 1d trend + Lips > Teeth > Jaw (aligned)
            if (close[i] > lips_aligned[i] and volume_confirmed and bullish_trend and 
                lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Lips + volume confirmation + bearish 1d trend + Lips < Teeth < Jaw (aligned)
            elif (close[i] < lips_aligned[i] and volume_confirmed and bearish_trend and 
                  lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Lips OR 1d trend turns bearish
            if close[i] < lips_aligned[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Lips OR 1d trend turns bullish
            if close[i] > lips_aligned[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals