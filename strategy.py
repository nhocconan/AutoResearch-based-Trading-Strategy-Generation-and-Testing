#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (3 SMAs) with volume confirmation and 1d EMA trend filter.
# The Alligator identifies trends when jaws (13 SMA), teeth (8 SMA), lips (5 SMA) are aligned.
# In bullish alignment (lips > teeth > jaws), go long on pullback to teeth with volume.
# In bearish alignment (lips < teeth < jaws), go short on pullback to teeth with volume.
# Uses 1d EMA50 for higher timeframe trend filter to avoid counter-trend trades.
# Session filter (08-20 UTC) reduces noise. Target: 20-50 trades/year (80-200 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Alligator calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 4h: SMAs of median price
    median_price_4h = (df_4h['high'] + df_4h['low']) / 2
    jaws = pd.Series(median_price_4h).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price_4h).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price_4h).rolling(window=5, min_periods=5).mean().values     # Green line
    
    # Align Alligator lines to 4h timeframe (already aligned to 4h bars)
    jaws_aligned = align_htf_to_ltf(prices, df_4h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_avg_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: lips > teeth > jaws (green > red > blue)
            bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaws_aligned[i])
            # Bearish alignment: lips < teeth < jaws (green < red < blue)
            bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaws_aligned[i])
            
            # Volume confirmation: current 4h volume > 1.5x average
            volume_ok = volume[i] > 1.5 * vol_avg_20_4h_aligned[i]
            
            # Long: Bullish alignment + price pulls back to teeth + above 1d EMA50 + volume
            if bullish and volume_ok:
                if (close[i] <= teeth_aligned[i] * 1.005 and  # Allow small overshoot
                    close[i] >= teeth_aligned[i] * 0.995 and
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
            # Short: Bearish alignment + price pulls back to teeth + below 1d EMA50 + volume
            elif bearish and volume_ok:
                if (close[i] >= teeth_aligned[i] * 0.995 and  # Allow small overshoot
                    close[i] <= teeth_aligned[i] * 1.005 and
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit: Price crosses jaws (alligator sleeping = trend over)
            if position == 1:
                if close[i] <= jaws_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= jaws_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_WilliamsAlligator_1dEMA50_Trend_Volume_Session"
timeframe = "4h"
leverage = 1.0