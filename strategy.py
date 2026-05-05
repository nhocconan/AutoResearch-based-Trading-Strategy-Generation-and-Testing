#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 Trend Filter and Volume Spike
# Long when Alligator jaws (13) < teeth (8) < lips (5) AND price > 1d EMA34 (strong uptrend) AND volume spike
# Short when Alligator jaws (13) > teeth (8) > lips (5) AND price < 1d EMA34 (strong downtrend) AND volume spike
# Williams Alligator identifies trend initiation and continuation with smoothed moving averages
# 1d EMA34 provides higher timeframe trend filter, reducing whipsaw in ranging markets
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing trends
# Works in bull (trend + Alligator alignment) and bear (trend continuation with volume confirmation)
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
    # Alligator: Jaw (13), Teeth (8), Lips (5) - all smoothed with specific offsets
    close_12h = close  # Using 12h close directly
    
    # Jaw: 13-period SMMA, shifted by 8 bars
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # Shifted by 8 bars
    jaw_values = jaw.values
    
    # Teeth: 8-period SMMA, shifted by 5 bars
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # Shifted by 5 bars
    teeth_values = teeth.values
    
    # Lips: 5-period SMMA, shifted by 3 bars
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # Shifted by 3 bars
    lips_values = lips.values
    
    # Volume confirmation on 12h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data for indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_values[i]) or 
            np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator aligned (jaw < teeth < lips) AND price > 1d EMA34 (strong uptrend) AND volume spike
            if (jaw_values[i] < teeth_values[i] and 
                teeth_values[i] < lips_values[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned (jaw > teeth > lips) AND price < 1d EMA34 (strong downtrend) AND volume spike
            elif (jaw_values[i] > teeth_values[i] and 
                  teeth_values[i] > lips_values[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses alignment (jaw > teeth OR teeth > lips) OR price crosses below 1d EMA34
            if (jaw_values[i] > teeth_values[i] or 
                teeth_values[i] > lips_values[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses alignment (jaw < teeth OR teeth < lips) OR price crosses above 1d EMA34
            if (jaw_values[i] < teeth_values[i] or 
                teeth_values[i] < lips_values[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals