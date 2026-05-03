#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator (jaw=13, teeth=8, lips=5 SMAs) identifies trend absence (alligator sleeping) vs presence (alligator awake).
# In ranging markets (jaws, teeth, lips intertwined): fade extremes at 1d Bollinger Bands (2,2) with volume confirmation.
# In trending markets (jaws, teeth, lips separated and ordered): trade pullbacks to 8-period SMA (teeth) in trend direction.
# Uses 1d timeframe for structure (Bollinger, EMA) and 12h for execution to minimize fee drag.
# Target: 12-30 trades/year (50-120 total over 4 years) to avoid fee drag while capturing BTC/ETH moves.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Bollinger Bands and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2) for mean reversion signals
    close_1d = df_1d['close'].values
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2.0 * std_20_1d
    lower_bb_1d = sma_20_1d - 2.0 * std_20_1d
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h: Jaw(13), Teeth(8), Lips(5) SMAs
    close_12h = df_12h['close'].values
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values  # Jaw (blue)
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values   # Teeth (red)
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values    # Lips (green)
    
    # Align Alligator components to 12h (no additional delay needed as SMAs use completed bar)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Alligator state: 
        # Sleeping (ranging): jaws, teeth, lips intertwined (|jaw-teeth| < 0.1% * price AND |teeth-lips| < 0.1% * price)
        # Awakening (trending): jaws, teeth, lips separated and ordered
        jaw_teeth_dist = abs(jaw_aligned[i] - teeth_aligned[i])
        teeth_lips_dist = abs(teeth_aligned[i] - lips_aligned[i])
        price_level = close[i]
        alligator_sleeping = (jaw_teeth_dist < 0.001 * price_level) and (teeth_lips_dist < 0.001 * price_level)
        
        if position == 0:
            if alligator_sleeping:
                # Ranging market: mean reversion at Bollinger extremes with volume confirmation
                if close[i] < lower_bb_aligned[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > upper_bb_aligned[i] and volume_spike:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: trade pullbacks to teeth (8 SMA) in trend direction
                # Trend direction: lips > teeth > jaw (bullish) OR lips < teeth < jaw (bearish)
                bullish_trend = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
                bearish_trend = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
                
                if bullish_trend and volume_spike:
                    # Long on pullback to teeth in uptrend
                    if close[i] <= teeth_aligned[i] * 1.002:  # Within 0.2% of teeth
                        signals[i] = 0.25
                        position = 1
                elif bearish_trend and volume_spike:
                    # Short on pullback to teeth in downtrend
                    if close[i] >= teeth_aligned[i] * 0.998:  # Within 0.2% of teeth
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: 
            # 1. In ranging market: price reaches upper Bollinger Band
            # 2. In trending market: trend reverses (lips crosses below teeth) OR price reaches 1.5x ATR above entry (simplified: close > upper BB)
            if alligator_sleeping:
                if close[i] > upper_bb_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Check for trend reversal: lips < teeth (bullish trend broken)
                if lips_aligned[i] < teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short:
            # 1. In ranging market: price reaches lower Bollinger Band
            # 2. In trending market: trend reverses (lips crosses above teeth) OR price reaches 1.5x ATR below entry (simplified: close < lower BB)
            if alligator_sleeping:
                if close[i] < lower_bb_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Check for trend reversal: lips > teeth (bearish trend broken)
                if lips_aligned[i] > teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals