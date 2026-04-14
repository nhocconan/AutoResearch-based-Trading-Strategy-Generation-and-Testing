#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d trend filter (EMA50) and volume confirmation.
# Alligator identifies trend direction and strength; avoids choppy markets.
# 1d EMA50 provides higher timeframe trend bias to avoid counter-trend trades.
# Volume > 1.5x average confirms institutional participation.
# Works in bull/bear as 1d EMA adapts to trend and Alligator filters chop.
# Target: 12-37 trades/year per symbol (50-150 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_len = 50
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Load 12h data ONCE for Alligator
    df_12h = get_htf_data(prices, '12h')
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_price = (df_12h['high'] + df_12h['low']) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Alligator signals: Lips > Teeth > Jaw = bullish; Lips < Teeth < Jaw = bearish
    # Alligator sleeping: all lines intertwined (choppy market)
    lips_above_teeth = lips_aligned > teeth_aligned
    teeth_above_jaw = teeth_aligned > jaw_aligned
    lips_below_teeth = lips_aligned < teeth_aligned
    teeth_below_jaw = teeth_aligned < jaw_aligned
    
    bullish_alligator = lips_above_teeth & teeth_above_jaw
    bearish_alligator = lips_below_teeth & teeth_below_jaw
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 13, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: bullish Alligator + above 1d EMA + volume
            if (bullish_alligator[i] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: bearish Alligator + below 1d EMA + volume
            elif (bearish_alligator[i] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish or price crosses below 1d EMA
            if bearish_alligator[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator turns bullish or price crosses above 1d EMA
            if bullish_alligator[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Alligator_EMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0