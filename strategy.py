#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA(34) trend filter + volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets
# Entries occur when Lips cross Teeth/Jaw in direction of 1d EMA(34) trend
# Volume spike (>2.0x 20-period EMA) confirms institutional participation
# Works in bull/bear markets by following 1d trend direction for entries
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: SMAs of median price (HL/2)
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars
    # Lips: 5-period SMA, shifted 3 bars
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams Alligator signals with 1d EMA trend filter
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Trend filter: price vs 1d EMA(34)
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Bullish alignment + price above 1d EMA34 + volume spike
            if bullish_aligned and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price below 1d EMA34 + volume spike
            elif bearish_aligned and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish alignment OR price below 1d EMA34
            if bearish_aligned or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish alignment OR price above 1d EMA34
            if bullish_aligned or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals