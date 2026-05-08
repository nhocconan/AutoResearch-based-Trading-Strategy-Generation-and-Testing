#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Trend and Volume Spike
# - Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) with future shift
# - Trend: Jaw > Teeth > Lips = bullish, Jaw < Teeth < Lips = bearish
# - Entry: Alligator aligned in trend direction + price crosses the middle line (Teeth) + volume spike
# - Works in bull/bear by using 1d trend filter to avoid counter-trend trades
# - Target: 15-30 trades/year to minimize fee drag on 12h timeframe

name = "12h_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator calculation on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Jaw (13-period SMMA, shifted 8 bars forward)
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth (8-period SMMA, shifted 5 bars forward)
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift forward 5 bars
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips (5-period SMMA, shifted 3 bars forward)
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift forward 3 bars
    lips[:3] = np.nan  # first 3 values invalid
    
    # Align Alligator lines to 12h timeframe (already on 12h, but need to align to lower timeframe if needed)
    # Since we're using 12h as primary timeframe, no alignment needed for Alligator
    # But we need to align the 1d trend filter
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Jaw > Teeth > Lips
            bullish = jaw[i] > teeth[i] > lips[i]
            # Bearish alignment: Jaw < Teeth < Lips
            bearish = jaw[i] < teeth[i] < lips[i]
            
            # Long: bullish alignment + price crosses above Teeth + volume spike
            long_cond = bullish and (close[i] > teeth[i]) and (close[i-1] <= teeth[i-1]) and volume_spike[i]
            
            # Short: bearish alignment + price crosses below Teeth + volume spike
            short_cond = bearish and (close[i] < teeth[i]) and (close[i-1] >= teeth[i-1]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Lips (or Alligator alignment breaks)
            if close[i] < lips[i] or not (jaw[i] > teeth[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Lips (or Alligator alignment breaks)
            if close[i] > lips[i] or not (jaw[i] < teeth[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals