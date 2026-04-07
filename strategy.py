#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour EMA pullback strategy with 4-hour trend filter and daily volume confirmation
# EMA(21) on 1h provides dynamic support/resistance. 4h EMA(50) determines trend direction.
# Daily volume filter ensures institutional participation. Designed for low frequency in 1h timeframe.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "1h_ema_pullback_4h_trend_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend
    close_4h = pd.Series(df_4h['close'].values)
    ema_4h = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average
    volume_1d = pd.Series(df_1d['volume'].values)
    vol_avg_1d = volume_1d.rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 1h EMA(21) for pullback entries
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_21[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume above daily average
        vol_filter = volume[i] > vol_avg_1d_aligned[i]
        
        # Trend filter: 4h EMA slope
        if i >= 51:
            ema_slope = ema_4h_aligned[i] - ema_4h_aligned[i-1]
            uptrend = ema_slope > 0
            downtrend = ema_slope < 0
        else:
            uptrend = ema_4h_aligned[i] > ema_4h_aligned[i-1] if i > 0 else False
            downtrend = ema_4h_aligned[i] < ema_4h_aligned[i-1] if i > 0 else False
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if trend changes or price crosses below EMA(21)
            if not uptrend or close[i] < ema_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit if trend changes or price crosses above EMA(21)
            if not downtrend or close[i] > ema_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: uptrend + price pulls back to EMA(21) + volume filter
            if uptrend and close[i] >= ema_21[i] * 0.995 and vol_filter:  # Allow 0.5% tolerance
                position = 1
                signals[i] = 0.20
            # Enter short: downtrend + price rallies to EMA(21) + volume filter
            elif downtrend and close[i] <= ema_21[i] * 1.005 and vol_filter:  # Allow 0.5% tolerance
                position = -1
                signals[i] = -0.20
    
    return signals