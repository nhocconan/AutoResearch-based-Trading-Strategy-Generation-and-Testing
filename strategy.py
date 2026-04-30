#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Long when price is above Alligator lips (green line) with 1w uptrend (close > 1w EMA50) and volume > 1.8x 20-bar avg.
# Short when price is below Alligator lips (green line) with 1w downtrend (close < 1w EMA50) and volume > 1.8x 20-bar avg.
# Exit when price crosses Alligator teeth (red line) in opposite direction.
# Uses Williams Alligator (SMAs of median price) for trend identification with strict volume confirmation.
# 1w EMA50 provides longer-term trend filter to avoid false signals in ranging markets.
# Timeframe: 1d, HTF: 1w as per experiment guidelines.

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Alligator Jaw (blue line) - 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Alligator Teeth (red line) - 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Alligator Lips (green line) - 5-period SMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price above Alligator lips, uptrend (close > 1w EMA50), volume spike
            if (curr_close > curr_lips and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price below Alligator lips, downtrend (close < 1w EMA50), volume spike
            elif (curr_close < curr_lips and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below Alligator teeth (trend weakening)
            if curr_close < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above Alligator teeth (trend weakening)
            if curr_close > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals