#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + 1w trend filter + volume confirmation
    # Long: Jaw < Teeth < Lips (bullish alignment) + price > Lips + 1w close > 1w EMA20 + volume > 1.5x 20-period avg
    # Short: Jaw > Teeth > Lips (bearish alignment) + price < Jaw + 1w close < 1w EMA20 + volume > 1.5x 20-period avg
    # Uses discrete sizing (0.25) to minimize fee drag
    # Target: 12-37 trades/year to stay within 12h optimal range (50-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (smoothed medians)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    
    # Williams Alligator lines (13,8,5 smoothed medians)
    jaw = pd.Series(median_12h).rolling(window=13, center=False).mean().rolling(window=8, center=False).mean().values
    teeth = pd.Series(median_12h).rolling(window=8, center=False).mean().rolling(window=5, center=False).mean().values
    lips = pd.Series(median_12h).rolling(window=5, center=False).mean().rolling(window=3, center=False).mean().values
    
    # Align to main timeframe (12h)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation (using 12h data)
    vol_avg_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_avg_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_12h[i]
        
        # Williams Alligator alignment
        bullish_alignment = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Trend filter: 1w close above/below EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        enter_long = bullish_alignment and (close[i] > lips_aligned[i]) and volume_confirmed and uptrend
        enter_short = bearish_alignment and (close[i] < jaw_aligned[i]) and volume_confirmed and downtrend
        
        # Exit conditions: reversal of alignment
        exit_long = position == 1 and not bullish_alignment
        exit_short = position == -1 and not bearish_alignment
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_williams_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0