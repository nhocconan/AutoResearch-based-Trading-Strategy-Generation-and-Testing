#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with volume confirmation and 12h trend filter
# Long when Alligator jaws (13) < teeth (8) < lips (5) + volume > 1.3x average + 12h trend up
# Short when Alligator jaws (13) > teeth (8) > lips (5) + volume > 1.3x average + 12h trend down
# Exit when Alligator lines cross or trend reverses
# Williams Alligator identifies trends with minimal lag, effective in both bull and bear markets
# Target: 15-25 trades/year on 4h timeframe with strong trend capture and low turnover

name = "4h_12h_alligator_volume_trend_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Williams Alligator components
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward  
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    def smoothed_moving_average(data, period):
        """Calculate Smoothed Moving Average (SMMA)"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        smma = np.full_like(data, np.nan)
        smma[period-1] = sma[period-1]
        for i in range(period, len(data)):
            smma[i] = (smma[i-1] * (period-1) + data[i]) / period
        return smma
    
    jaw = smoothed_moving_average(close, 13)  # 13-period SMMA
    teeth = smoothed_moving_average(close, 8)   # 8-period SMMA
    lips = smoothed_moving_average(close, 5)    # 5-period SMMA
    
    # Apply forward shifts as per Williams Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that go out of bounds
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):  # Start after EMA34 warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Trend filter: price relative to 12h EMA34
        is_uptrend = close[i] > ema_34_12h_aligned[i]
        is_downtrend = close[i] < ema_34_12h_aligned[i]
        
        # Alligator alignment conditions
        # Bullish alignment: Jaw < Teeth < Lips (alligator sleeping -> waking up to eat)
        bullish_alignment = (jaw_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < lips_shifted[i])
        # Bearish alignment: Jaw > Teeth > Lips (alligator sleeping -> waking up to eat downside)
        bearish_alignment = (jaw_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > lips_shifted[i])
        
        # Entry conditions
        long_entry = bullish_alignment and volume_filter and is_uptrend
        short_entry = bearish_alignment and volume_filter and is_downtrend
        
        # Exit conditions: Alligator lines cross or trend reverses
        long_exit = (jaw_shifted[i] >= teeth_shifted[i]) or (not is_uptrend)
        short_exit = (jaw_shifted[i] <= teeth_shifted[i]) or (not is_downtrend)
        
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