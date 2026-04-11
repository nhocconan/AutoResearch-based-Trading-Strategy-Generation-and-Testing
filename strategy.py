#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + volume confirmation + 1d/1w trend filter
# Long when Alligator mouth opens upward (green line > red > blue) + volume > 1.5x avg + 1d trend up
# Short when Alligator mouth opens downward (green < red < blue) + volume > 1.5x avg + 1d trend down
# Exit when Alligator lines intertwine (mouth closes) or trend reverses
# Designed for 20-40 trades/year on 4h timeframe with strong trend capture and low turnover

name = "4h_1d_williams_alligator_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator (13,8,5) with offsets (8,5,3)
    # Jaw (blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (red): 8-period SMMA, shifted 5 bars forward  
    # Lips (green): 5-period SMMA, shifted 3 bars forward
    close_series = pd.Series(close)
    smma_13 = close_series.rolling(window=13, min_periods=13).mean()
    smma_8 = close_series.rolling(window=8, min_periods=8).mean()
    smma_5 = close_series.rolling(window=5, min_periods=5).mean()
    
    jaw = smma_13.shift(8).values  # blue line
    teeth = smma_8.shift(5).values  # red line
    lips = smma_5.shift(3).values   # green line
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after longest SMMA period
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Alligator mouth direction
        mouth_open_up = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])   # Green > Red > Blue
        mouth_open_down = (lips[i] < teeth[i]) and (teeth[i] < jaw[i]) # Green < Red < Blue
        
        # Entry conditions
        long_entry = mouth_open_up and volume_filter and is_uptrend
        short_entry = mouth_open_down and volume_filter and is_downtrend
        
        # Exit conditions: mouth closes (lines intertwine) or trend reverses
        mouth_closed = not (mouth_open_up or mouth_open_down)  # Lines are intertwined
        long_exit = mouth_closed or (not is_uptrend)
        short_exit = mouth_closed or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals