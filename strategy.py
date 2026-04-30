#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Williams Alligator consists of three SMAs (Jaw:13, Teeth:8, Lips:5) smoothed.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 2.0x 20-bar average.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 2.0x 20-bar average.
# Exit when Alligator alignment breaks (Lips crosses Teeth or Jaw).
# Alligator identifies trend absence (sleeping) vs trend presence (awakening with alignment).
# 1d EMA50 filters for dominant intermediate-term trend to avoid counter-trend entries.
# Volume confirmation ensures institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components (SMAs with smoothing)
    # Jaw: 13-period SMA, smoothed by 8 periods
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().values
    # Teeth: 8-period SMA, smoothed by 5 periods
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().values
    # Lips: 5-period SMA, smoothed by 3 periods
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 8, 5)  # warmup for Alligator components and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish alignment (Lips > Teeth > Jaw), uptrend (price > 1d EMA50), volume confirmation
            if (curr_lips > curr_teeth and curr_teeth > curr_jaw and 
                curr_close > ema_50_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment (Lips < Teeth < Jaw), downtrend (price < 1d EMA50), volume confirmation
            elif (curr_lips < curr_teeth and curr_teeth < curr_jaw and 
                  curr_close < ema_50_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Bullish alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if curr_lips <= curr_teeth or curr_teeth <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Bearish alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if curr_lips >= curr_teeth or curr_teeth >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals