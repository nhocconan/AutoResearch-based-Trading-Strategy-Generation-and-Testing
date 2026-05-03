#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses weekly EMA50 for primary trend direction and daily Donchian channels for entry timing.
# Volume confirmation ensures breakouts have institutional participation. Designed to work
# in both bull and bear markets by only taking breakouts in the direction of the weekly trend.
# Target: 15-25 trades/year to minimize fee drag while capturing high-probability trend continuations.

name = "1d_Donchian20_1wEMA50_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 20-period Donchian channels from daily data
    # Upper channel = highest high of last 20 days
    # Lower channel = lowest low of last 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume regime: current 1d volume > 1.8x 20-period MA (moderate threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(upper) or np.isnan(lower) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Entry conditions
        # Long: break above upper Donchian with volume spike and above weekly EMA50
        long_entry = (close[i] > upper) and vol_spike and (close[i] > ema_trend)
        # Short: break below lower Donchian with volume spike and below weekly EMA50
        short_entry = (close[i] < lower) and vol_spike and (close[i] < ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on close below weekly EMA50 (trend change)
            if close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on close above weekly EMA50 (trend change)
            if close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals