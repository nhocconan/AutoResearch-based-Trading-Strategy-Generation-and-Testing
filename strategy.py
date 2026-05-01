#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1w EMA50 trend filter + volume spike confirmation
# Williams Alligator identifies trend via jaw/teeth/lips SMAs; trade in trend direction
# 1w EMA50 ensures alignment with weekly supertrend to avoid counter-trend whipsaws
# Volume spike > 2.0x 20-period EMA confirms institutional participation
# Designed for low trade frequency: ~12-30 trades/year per symbol with 0.25 sizing
# Works in both bull and bear markets by only taking trend-aligned Alligator signals

name = "6h_WilliamsAlligator_1wEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: jaw (13), teeth (8), lips (5) SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for Alligator calculation
    start_idx = 13
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50: bullish if price > EMA50, bearish if price < EMA50
        bullish_bias = close[i] > ema_50_1w_aligned[i]
        bearish_bias = close[i] < ema_50_1w_aligned[i]
        
        # Williams Alligator signals: 
        # Bullish: lips > teeth > jaw (green alignment)
        # Bearish: lips < teeth < jaw (red alignment)
        bullish_alligator = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alligator = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and bullish_alligator:
                # Long: bullish Alligator alignment with volume spike
                if volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias and bearish_alligator:
                # Short: bearish Alligator alignment with volume spike
                if volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No clear signal or no volume confirmation
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks or price crosses 1w EMA50
            if not bullish_alligator or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks or price crosses 1w EMA50
            if not bearish_alligator or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals